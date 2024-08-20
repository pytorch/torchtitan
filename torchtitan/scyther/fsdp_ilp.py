"""
To use the HiGHS solver, you need to install it first and pass the path to the solver.
Follow instructions here: https://ergo-code.github.io/HiGHS/dev/interfaces/cpp/

Some example commands to run:
    python fsdp_ilp.py --in_file=GPT_modules_info.json --memory_budget=3
    python fsdp_ilp.py --in_file=GPT_modules_info.json --memory_budget=4 --verbose
    python fsdp_ilp.py --in_file=GPT_modules_info.json --memory_budget=4 --verbose \
        --fsdp_units GPT.transformer.h.0 GPT.transformer.h.1 GPT.transformer.h.2 \
        GPT.transformer.h.3 GPT.transformer.h.4 GPT.transformer.h.5
"""

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from comm_analysis import NCCL_COLL
from commtime_estimator import get_collective_latency_bandwidth

from ilp_utils import (
    display_bytes,
    get_peak_memory_runtime_no_ac_fsdp,
    Graph,
    parse_input,
)
from pulp import (
    COIN_CMD,
    HiGHS_CMD,
    lpDot,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    PULP_CBC_CMD,
    value,
)

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)

# Create a stream handler to print log messages to the terminal
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add the handler to the logger
logger.addHandler(handler)


@dataclass
class CommParams:
    latency: int
    bandwith: int


def fsdp_milp(
    graph: Graph,
    world_size: int,
    comm_params: Dict[str, CommParams],
    memory_budget: int,
    solver: COIN_CMD,
    fsdp_units: List[str] = None,
    selective_ac: bool = False,
    verbose: bool = False,
) -> None:
    """
    MILP to decide FSDP units, AC units and how much memory to discard.
    Objective: minimize recomputation time.
    Constratint: memory budget (in bytes).
    """

    # TODO: link doc with formulation
    # TODO: add sac functionality

    num_nodes = len(graph.nodes)
    BIG_M = 1000
    MEM_MULTIPLIER = 2**30

    # Create a MILP problem
    prob = LpProblem("FSDP", LpMinimize)

    # Create decision variables
    x = LpVariable.matrix("x", list(range(num_nodes)), 0, 1, LpInteger)
    p = LpVariable.matrix("p", list(range(num_nodes)), 0)
    g = LpVariable.matrix("g", list(range(num_nodes)), 0)
    a = LpVariable.matrix("a", list(range(num_nodes)), 0)
    m = LpVariable.matrix("m", list(range(num_nodes)), 0)
    max_m = LpVariable("max_m", 0)
    max_p = LpVariable("max_p", 0)
    ag = LpVariable.matrix("ag", list(range(num_nodes)), 0)
    t0 = LpVariable.matrix("t0", list(range(num_nodes)), 0)
    fw_ag = LpVariable.matrix("fw_ag", list(range(num_nodes)), 0)
    t1 = LpVariable.matrix("t1", list(range(num_nodes)), 0)
    bw_ag = LpVariable.matrix("bw_ag", list(range(num_nodes)), 0)
    rs = LpVariable.matrix("rs", list(range(num_nodes)), 0)
    t2 = LpVariable.matrix("t2", list(range(num_nodes)), 0)
    bw_rs = LpVariable.matrix("bw_rs", list(range(num_nodes)), 0)
    t3 = LpVariable.matrix("t3", list(range(num_nodes)), 0)
    fw_e = LpVariable.matrix("fw_e", list(range(num_nodes)), 0)
    t4 = LpVariable.matrix("t4", list(range(num_nodes)), 0)
    bw_e = LpVariable.matrix("bw_e", list(range(num_nodes)), 0)

    # Add constraints
    P_1 = graph.nodes[0]["param_per_module"] / MEM_MULTIPLIER
    G_1 = graph.nodes[0]["grad_per_module"] / MEM_MULTIPLIER
    # [Constraint] Root module is always an FSDP unit
    prob += x[0] == 1

    # [Constraint] No nested FSDP unit
    if fsdp_units:
        fsdp_units = set(fsdp_units)
        for i in range(1, num_nodes):
            if graph.nodes[i]["fqn"] in fsdp_units:
                prob += x[i] == 1
            else:
                prob += x[i] == 0
    else:
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes):
                if graph.ad_matrix[i][j] == 1:
                    prob += x[i] + x[j] <= 1

    # [Constraint] Express parameter taken care of by each module for FSDP
    for i in range(1, num_nodes):
        P_i = graph.nodes[i]["param_per_module"] / MEM_MULTIPLIER
        prob += p[i] == P_i * x[i]
    prob += p[0] == P_1 - lpSum(p[1:])

    # [Constraint] Express gradient taken care of by each module for FSDP
    for i in range(1, num_nodes):
        G_i = graph.nodes[i]["grad_per_module"] / MEM_MULTIPLIER
        prob += g[i] == G_i * x[i]
    prob += g[0] == G_1 - lpSum(g[1:])

    # [Constraint] Express the total amount memory at each module
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"] / MEM_MULTIPLIER
        coeff = np.zeros(num_nodes)
        for j in range(num_nodes):
            if graph.ad_matrix[j][i] == 1:
                coeff[j] = 1
        prob += (
            m[i] == (P_1 + TG_i) / world_size + lpDot(p, coeff) + lpDot(g, coeff) + a[i]
        )

    # [Constraint] Express total activation memory in the backward pass
    for i in range(num_nodes):
        AG_i = graph.nodes[i]["act_grad_per_module"] / MEM_MULTIPLIER
        TA_i = graph.nodes[i]["act_total"] / MEM_MULTIPLIER
        prob += a[i] == TA_i + AG_i

    # [Constraint] Express peak memory
    for i in range(num_nodes):
        prob += max_m >= m[i]

    # [Constraint] Express maximum FSDP shard
    for i in range(num_nodes):
        prob += max_p >= p[i]

    # [Constraint] Respect memory budget
    prob += max_m + 2 * max_p <= memory_budget

    # [Constraint] Express the all gather communication time of each FSDP unit
    comm_model = comm_params["all_gather"]
    for i in range(num_nodes):
        prob += ag[i] == comm_model.latency + p[i] * (
            MEM_MULTIPLIER / comm_model.bandwith
        )

    # [Constraint] Express the reduce scatter communication time of each FSDP unit
    comm_model = comm_params["reduce_scatter"]
    for i in range(num_nodes):
        prob += rs[i] == comm_model.latency + g[i] * (
            MEM_MULTIPLIER / comm_model.bandwith
        )

    # [Constraint] Express the forward prefetch all gather communication time
    prob += t0[num_nodes - 1] == ag[num_nodes - 1]
    for i in range(1, num_nodes - 1):
        prob += t0[i] <= t0[i + 1] + BIG_M * x[i]
        prob += t0[i] >= t0[i + 1] - BIG_M * x[i]
        prob += t0[i] <= ag[i] + BIG_M * (1 - x[i])
        prob += t0[i] >= ag[i] - BIG_M * (1 - x[i])
    prob += fw_ag[num_nodes - 1] == 0
    for i in range(num_nodes - 1):
        prob += fw_ag[i] <= BIG_M * x[i]
        prob += fw_ag[i] <= t0[i + 1]
        prob += fw_ag[i] >= t0[i + 1] - BIG_M * (1 - x[i])

    # [Constraint] Express the backward prefetch all gather communication time
    # this is the index of modules in the backward pre order
    o1 = [graph.name2node[fqn]["index"] for fqn in reversed(graph.fw_post_order)]
    prob += t1[o1[num_nodes - 1]] == ag[o1[num_nodes - 1]]
    for k in range(1, num_nodes - 1):
        i = o1[k]
        i_next = o1[k + 1]
        prob += t1[i] <= t1[i_next] + BIG_M * x[i]
        prob += t1[i] >= t1[i_next] - BIG_M * x[i]
        prob += t1[i] <= ag[i] + BIG_M * (1 - x[i])
        prob += t1[i] >= ag[i] - BIG_M * (1 - x[i])
    prob += bw_ag[o1[num_nodes - 1]] == 0
    for k in range(1, num_nodes - 1):
        i = o1[k]
        i_next = o1[k + 1]
        prob += bw_ag[i] <= BIG_M * x[i]
        prob += bw_ag[i] <= t1[i_next]
        prob += bw_ag[i] >= t1[i_next] - BIG_M * (1 - x[i])

    # [Constraint] Express the previous module's reduce scatter communication time
    prob += t2[num_nodes - 1] == rs[num_nodes - 1]
    for i in range(1, num_nodes - 1):
        prob += t2[i] <= t2[i + 1] + BIG_M * x[i]
        prob += t2[i] >= t2[i + 1] - BIG_M * x[i]
        prob += t2[i] <= rs[i] + BIG_M * (1 - x[i])
        prob += t2[i] >= rs[i] - BIG_M * (1 - x[i])
    prob += bw_rs[num_nodes - 1] == 0
    for i in range(num_nodes - 1):
        prob += bw_rs[i] <= BIG_M * x[i]
        prob += bw_rs[i] <= t2[i + 1]
        prob += bw_rs[i] >= t2[i + 1] - BIG_M * (1 - x[i])

    # [Constraint] Express the exposed computation time in the forward pass
    for i in range(1, num_nodes):
        FCP_i = graph.nodes[i]["fw_runtime_per_module"]
        prob += t3[i] >= fw_ag[i] - FCP_i
        prob += fw_e[i] <= BIG_M * x[i]
        prob += fw_e[i] <= t3[i]
        prob += fw_e[i] >= t3[i] - BIG_M * (1 - x[i])
    prob += fw_e[0] == 0

    # [Constraint] Express the exposed computation time in the backward pass
    for i in range(1, num_nodes):
        BCP_i = graph.nodes[i]["bw_runtime_per_module"]
        prob += t4[i] >= bw_ag[i] + bw_rs[i] - BCP_i
        prob += bw_e[i] <= BIG_M * x[i]
        prob += bw_e[i] <= t4[i]
        prob += bw_e[i] >= t4[i] - BIG_M * (1 - x[i])
    prob += bw_e[0] == 0

    # Set Objeictive
    prob += lpSum(fw_e[1:]) + lpSum(bw_e[1:]) + ag[0] + rs[0] + fw_ag[0] + bw_rs[0]

    # Solve
    start_time = time.time()
    status = prob.solve(solver)
    end_time = time.time()
    logger.info(f"Solver completed in {round(end_time - start_time, 2)} sec")
    if status != 1:
        logger.info(f"Solver failed to find a solution: {LpStatus[status]}")
        return

    # Print solution
    fsdp_decisions = set()
    for i in range(num_nodes):
        if round(value(x[i]) if x[i] else 0) == 1:
            fsdp_decisions.add(graph.nodes[i]["fqn"])
    peak_mem = (max_m.varValue + 2 * max_p.varValue) * MEM_MULTIPLIER
    obj = round(value(prob.objective), 4)

    logger.info(
        f"On {world_size} GPUs\n"
        + f"  FSDP units are {fsdp_decisions}\n"
        + f"  peak memory is {display_bytes(peak_mem, 'GiB')}\n"
        + f"  total exposed computation time is {obj} ms"
    )

    if verbose:
        logger.info("\n\n --------- DETAILS ---------")
        for i in range(num_nodes):
            x_i = value(x[i]) if x[i] else 0
            p_i = p[i].varValue * MEM_MULTIPLIER
            g_i = g[i].varValue * MEM_MULTIPLIER
            a_i = a[i].varValue * MEM_MULTIPLIER
            m_i = m[i].varValue * MEM_MULTIPLIER
            ag_i = ag[i].varValue if ag[i] else 0
            fw_ag_i = fw_ag[i].varValue if fw_ag[i] else 0
            bw_ag_i = bw_ag[i].varValue if bw_ag[i] else 0
            rs_i = rs[i].varValue if rs[i] else 0
            bw_rs_i = bw_rs[i].varValue if bw_rs[i] else 0
            FCP_i = graph.nodes[i]["fw_runtime_per_module"]
            BCP_i = graph.nodes[i]["bw_runtime_per_module"]
            fw_e_i = fw_e[i].varValue if fw_e[i] else 0
            bw_e_i = bw_e[i].varValue if bw_e[i] else 0
            logger.info(
                ("FSDP" if round(x_i) == 1 else "    ")
                + f" {graph.nodes[i]['fqn']:<40}: "
                + f"p_i = {display_bytes(p_i, 'GiB'):<10} "
                + f"g_i = {display_bytes(g_i, 'GiB'):<10} "
                + f"a_i = {display_bytes(a_i, 'GiB'):<10} "
                + f"m_i = {display_bytes(m_i, 'GiB'):<10} "
                + f"ag_i = {round(ag_i, 2):5.2f} ms "
                + f"fw_ag_i = {round(fw_ag_i, 2):5.2f} ms "
                + f"bw_ag_i = {round(bw_ag_i, 2):5.2f} ms "
                + f"rs_i = {round(rs_i, 2):5.2f} ms "
                + f"bw_rs_i = {round(bw_rs_i, 2):5.2f} ms "
                + f"FCP_i = {FCP_i:8.2f} ms "
                + f"BCP_i = {BCP_i:8.2f} ms "
                + f"fw_e_i = {round(fw_e_i, 2):5.2f} ms "
                + f"bw_e_i = {round(bw_e_i, 2):5.2f} ms "
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--in_file",
        help="Input file with module information",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--solver",
        help="Solver for MILP",
        required=False,
        choices=["CBC", "HiGHS"],
        default="CBC",
        type=str,
    )

    parser.add_argument(
        "--solver_path",
        help="Path to solver binary",
        required=False,
        type=str,
        default="",
    )

    parser.add_argument(
        "--world_size",
        help="Number of GPUs",
        required=False,
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_gpus_per_node",
        help="Number of GPUs per node",
        required=False,
        type=int,
        default=8,
    )

    parser.add_argument(
        "--memory_budget",
        help="Memory budget in GiB",
        required=False,
        type=float,
        default=70,
    )

    parser.add_argument(
        "--verbose",
        help="Verbosity level",
        action="store_true",
    )

    parser.add_argument(
        "--solver_msg",
        help="Turn on/off solver messages",
        action="store_true",
    )

    parser.add_argument("--fsdp_units", "--names-list", nargs="+", default=[])

    args = parser.parse_args()
    return args


def main():
    # parse the input
    args = parse_args()

    # communication model
    all_gather_latency, all_gather_bw = get_collective_latency_bandwidth(
        NCCL_COLL.ALL_GATHER, args.world_size, args.num_gpus_per_node
    )
    reduce_scatter_latency, reduce_scatter_bw = get_collective_latency_bandwidth(
        NCCL_COLL.ALL_GATHER, args.world_size, args.num_gpus_per_node
    )
    comm_params = {
        "all_gather": CommParams(all_gather_latency, all_gather_bw),
        "reduce_scatter": CommParams(reduce_scatter_latency, reduce_scatter_bw),
    }

    # get the json file by running `python aggregate_stats.py`
    graph = parse_input(args.in_file)

    # setup and solve the problem
    solver = PULP_CBC_CMD(msg=args.solver_msg)
    if args.solver == "HiGHS":
        try:
            if args.solver_path:
                solver = HiGHS_CMD(path=args.solver_path, msg=args.solver_msg)
            else:
                solver = HiGHS_CMD(msg=args.solver_msg)
        except Exception:
            logger.error("HiGHS solver not found. Using CBC instead.")

    # get the memory utilization without fsdp
    peak_mem, compute_time = get_peak_memory_runtime_no_ac_fsdp(graph)
    logger.info(
        "On a single GPU without AC \n"
        + f"  peak memory is {display_bytes(peak_mem, 'GiB')}\n"
        + f"  compute time is {round(compute_time, 2)} ms\n"
        + "---" * 20
    )
    logger.info(
        "On a single GPU\n"
        + f"  peak memory is {display_bytes(peak_mem, 'GiB')}\n"
        + f"  compute time is {round(compute_time, 2)} ms\n"
        + "---" * 20
    )

    fsdp_milp(
        graph,
        world_size=args.world_size,
        comm_params=comm_params,
        memory_budget=args.memory_budget,
        solver=solver,
        fsdp_units=args.fsdp_units,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

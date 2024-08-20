"""
To use the HiGHS solver, you need to install it first and pass the path to the solver.
Follow instructions here: https://ergo-code.github.io/HiGHS/dev/interfaces/cpp/

Command to run:
    python sac_ilp.py --in_file=GPT_modules_info.json --memory_budget=6
    python sac_ilp.py --in_file=GPT_modules_info.json --memory_budget=6 \
        --solver=HiGHS --solver_path=/home/xuanzh/local/HiGHS/build/bin/highs
"""

import argparse
import json
import logging
import time
from typing import List

import numpy as np
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


def sac_milp(
    graph: Graph,
    memory_budget: int,
    solver: COIN_CMD,
    ac_units: List[str] = None,
    verbose: bool = False,
) -> None:
    """
    MILP to decide which modules to AC and how much memory to discard.
    Objective: minimize recomputation time.
    Constratint: memory budget (in bytes).
    #TODO: link doc with formulation
    """
    num_nodes = len(graph.nodes)
    M = 10**2  # note: numerical issue may occur if M is too big
    MEM_MULTIPLIER = 2**30

    # Create a MILP problem
    prob = LpProblem("SAC", LpMinimize)

    # Create decision variables
    y = LpVariable.matrix("y", list(range(num_nodes)), 0, 1, LpInteger)
    r = LpVariable.matrix("r", list(range(num_nodes)), 0, 1)
    d = LpVariable.matrix("d", list(range(num_nodes)), 0)
    a = LpVariable.matrix("a", list(range(num_nodes)), 0)
    m = LpVariable.matrix("m", list(range(num_nodes)), 0)
    rcp = LpVariable.matrix("rcp", list(range(num_nodes)), 0)
    rct = LpVariable.matrix("rct", list(range(num_nodes)), 0)
    max_m = LpVariable("max_m", 0)

    # Add constraints
    # [Constraint] User specified AC units
    if ac_units:
        ac_units = set(ac_units)
        for i in range(num_nodes):
            if not graph.nodes[i]["fqn"] in ac_units:
                prob += y[i] == 0

    # [Constraint] No nested AC units
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if graph.ad_matrix[i][j] == 1:
                prob += y[i] + y[j] <= 1

    # [Constraint] Do not AC leaf modules
    for i in range(num_nodes):
        if graph.nodes[i]["is_leaf"]:
            prob += y[i] == 0

    # [Constraint] Express amount of discarded activation memory
    for i in range(num_nodes):
        ACM_i = graph.nodes[i]["ac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        prob += d[i] == ACM_i * r[i] - (ACM_i - IA_i) * y[i]

    # [Constraint] Express total activation memory in the backward pass
    for i in range(num_nodes):
        AG_i = graph.nodes[i]["act_grad_per_module"] / MEM_MULTIPLIER
        TA_i = graph.nodes[i]["act_total"] / MEM_MULTIPLIER
        ACM_i = graph.nodes[i]["ac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        # related to discarded amount of memory
        pos = graph.nodes[i]["pos_fw_post_order"]
        coeff = np.zeros(num_nodes)
        for p in range(pos):
            j = graph.name2node[graph.fw_post_order[p]]["index"]
            coeff[j] = 1
        if graph.nodes[i]["is_leaf"]:
            continue
        prob += a[i] + lpDot(coeff, d) == TA_i + AG_i

    # [Constraint] Express the total amount of memory at each module
    P_1 = graph.nodes[0]["param_per_module"] / MEM_MULTIPLIER
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"] / MEM_MULTIPLIER
        prob += m[i] - a[i] == P_1 + TG_i

    # [Constraint] Express peak memory
    for i in range(num_nodes):
        prob += max_m >= m[i]

    # [Constraint] Ensure correctness of r_i
    for i in range(num_nodes):
        prob += y[i] >= r[i]
        if graph.nodes[i]["is_leaf"]:
            continue
        ACM_i = graph.nodes[i]["ac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        prob += r[i] >= (ACM_i - IA_i) / ACM_i * y[i]

    # [Constraint] Express percentage of recomputation time
    for i in range(num_nodes):
        for s in range(graph.nodes[i]["n_segments"]):
            slope = graph.nodes[i]["slopes"][s]
            intercept = graph.nodes[i]["intercepts"][s]
            prob += rcp[i] - slope * r[i] >= intercept

    # [Constraint] Express recomputation time rec_i = y_i * (rep_i * FCP_i)
    for i in range(num_nodes):
        ACT_i = graph.nodes[i]["ac_runtime"]
        prob += rct[i] <= M * y[i]
        prob += rct[i] <= ACT_i * rcp[i]
        prob += rct[i] >= ACT_i * rcp[i] - M * (1 - y[i])

    # [Constraint] Peak memory should be below budget
    prob += max_m <= memory_budget

    # Set Objeictive
    prob += lpSum(rct)

    # Solve
    start_time = time.time()
    status = prob.solve(solver)
    end_time = time.time()
    logger.info(f"Solver completed in {round(end_time - start_time, 2)} sec")
    if status != 1:
        logger.info(f"Solver failed to find a solution: {LpStatus[status]}")
        return

    # Print solution
    ac_decisions = {}
    for i in range(num_nodes):
        if round(y[i].varValue) == 1:
            ac_decisions[graph.nodes[i]["fqn"]] = round(r[i].varValue, 4)
    logger.info(f"AC decisions are {json.dumps(ac_decisions, indent=2)}")
    logger.info(f"recomputation time is {round(value(prob.objective), 2)} ms")
    peak_mem = max_m.varValue * MEM_MULTIPLIER
    logger.info(f"peak memory is below {display_bytes(peak_mem, 'GiB')}")

    if verbose:
        logger.info("\n\n --------- DETAILS ---------")
        for i in range(num_nodes):
            if graph.nodes[i]["is_leaf"]:
                continue
            y_i = y[i].varValue
            r_i = r[i].varValue
            d_i = d[i].varValue * MEM_MULTIPLIER
            a_i = a[i].varValue * MEM_MULTIPLIER
            m_i = m[i].varValue * MEM_MULTIPLIER
            rcp_i = rcp[i].varValue if rcp[i].varValue else 0
            rct_i = rct[i].varValue
            logger.info(
                ("AC" if round(y_i) == 1 else "  ")
                + f" {graph.nodes[i]['fqn']:<40}: "
                + f"r_i = {r_i:.4f} "
                + f"a_i = {display_bytes(a_i, 'GiB'):<10} "
                + f"d_i = {display_bytes(d_i, 'GiB'):<10} "
                + f"m_i = {display_bytes(m_i, 'GiB'):<10} "
                + f"rcp_i = {rcp_i:8.4f} "
                + f"rct_i = {rct_i:8.4f} "
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

    parser.add_argument("--ac_units", "--names-list", nargs="+", default=[])

    args = parser.parse_args()
    return args


def main():
    # parse the input
    args = parse_args()

    # get the json file by running `python aggregate_stats.py`
    graph = parse_input(args.in_file)

    # get the memory utilization without ac
    peak_mem, compute_time = get_peak_memory_runtime_no_ac_fsdp(graph)
    logger.info(
        "On a single GPU without AC \n"
        + f"  peak memory is {display_bytes(peak_mem, 'GiB')}\n"
        + f"  compute time is {round(compute_time, 2)} ms\n"
        + "---" * 20
    )

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
    sac_milp(
        graph,
        memory_budget=args.memory_budget,
        solver=solver,
        ac_units=args.ac_units,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

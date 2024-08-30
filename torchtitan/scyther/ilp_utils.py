import json
from typing import Dict, List, Tuple

import numpy as np
from aggregate_stats import ModStats


class Node(ModStats):
    index: int = 0  # index according to forward pre-order
    pos_fw_post_order: int = 0  # index according to forward post-order


class Graph:
    def __init__(self, name: str, n: int) -> None:
        self.name: str = name
        self.nodes: List[Node] = []
        self.name2node: Dict[str, Node] = {}
        self.ad_matrix = np.zeros((n, n))
        self.fw_post_order: List[str] = []

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.name2node[node["fqn"]] = node


def parse_input(filename: str) -> Graph:
    with open(filename, "r") as f:
        module_info = json.load(f)

    # assertion and number of nodes
    assert len(module_info["modstats"]) == len(module_info["fw_pre_order"])
    n_nodes = len(module_info["modstats"])

    # create graph
    model_name = filename.split("_")[0]
    g = Graph(model_name, n_nodes)
    g.fw_post_order = module_info["fw_post_order"]

    # sort the modules by pre-order and add them to the graph
    module_info["modstats"] = sorted(
        module_info["modstats"],
        key=lambda x: module_info["fw_pre_order"].index(x["fqn"]),
    )
    for i, mod_info in enumerate(module_info["modstats"]):
        node: Node = mod_info
        node["index"] = i
        node["pos_fw_post_order"] = g.fw_post_order.index(node["fqn"])
        g.add_node(node)

    # set up ancestor-descendant matrix
    def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
        # if name_descendant is a submodule of name_ancestor, or if they are the same
        return (
            name_descendant == name_ancestor or name_ancestor + "." in name_descendant
        )

    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if is_self_or_submodule(g.nodes[j]["fqn"], g.nodes[i]["fqn"]):
                g.ad_matrix[i][j] = 1
            else:
                break

    return g


def display_bytes(b: int, unit: str = "B") -> str:
    """
    return a string that represent the number of bytes in a desired unit
    """
    if unit == "KiB":
        return f"{b/2**10:.2f} KiB"
    if unit == "MiB":
        return f"{b/2**20:.2f} MiB"
    if unit == "GiB":
        return f"{b/2**30:.2f} GiB"
    return f"{b:.2f} bytes"


def get_peak_memory_runtime_no_ac_fsdp(graph: Graph) -> Tuple[int, float]:
    """Get the peak memory without FSDP"""
    P_1 = graph.nodes[0]["param_per_module"]
    num_nodes = len(graph.nodes)
    peak_mem = 0
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"]
        AG_i = graph.nodes[i]["act_grad_per_module"]
        TA_i = graph.nodes[i]["act_total"]
        peak_mem = max(peak_mem, P_1 + TG_i + AG_i + TA_i)
    compute_time = (
        graph.nodes[0]["fw_runtime_per_module"]
        + graph.nodes[0]["bw_runtime_per_module"]
    )
    return (peak_mem, compute_time)

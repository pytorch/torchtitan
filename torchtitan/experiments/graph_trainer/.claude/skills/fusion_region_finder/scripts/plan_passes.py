# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turn a wish list of lift passes into a fusion plan that runs as listed.

Usage: ``python plan_passes.py <capture dir> lift_fusion_region_g006 lift_fusion_region_g020 ...``

Replays the passes in the given order on the capture's ``pre_inductor`` graph, as
``config.compile.extra_passes`` would run them: each pass takes the safe sites the
passes before it left. Passes with nothing left are dropped (a pass that lifts
nothing raises at runtime). Prints the plan as one comma-separated line, and per pass
the sites it lifts, on stderr.
"""

import argparse
import importlib
import os
import sys

from lift_regions import find_matches, lift_hazard, load_dump_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="capture directory with fusion_regions/")
    ap.add_argument("passes", nargs="+", help="lift pass names in priority order")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.root))
    fusion_regions = importlib.import_module("fusion_regions")
    folders = {
        pass_name: folder for folder, pass_name in fusion_regions.PASSES.values()
    }
    nodes = load_dump_graph(
        os.path.join(args.root, "graph_dump", "pre_inductor_nodes.jsonl.gz")
    )
    taken, plan = set(), []
    for pass_name in args.passes:
        if pass_name not in folders:
            raise ValueError(f"{pass_name} is not a lift pass of {args.root}")
        module = importlib.import_module(
            f"fusion_regions.{folders[pass_name]}.lift_pass"
        )
        sites = find_matches(nodes, module.MATCHERS, claimed=set(taken))
        safe = [site for site in sites if not lift_hazard(site)]
        print(
            f"{pass_name}: {len(safe)} site(s)"
            + ("" if safe else " -- dropped, an earlier pass took them"),
            file=sys.stderr,
        )
        if safe:
            plan.append(pass_name)
            taken |= {n for site in safe for n in site}
    print(",".join(plan))


if __name__ == "__main__":
    main()

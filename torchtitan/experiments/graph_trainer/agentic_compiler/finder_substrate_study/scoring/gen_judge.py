# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pool one model's regions into a blind judge sandbox and write the judge prompt.

    python gen_judge.py <model>   # -> <ROOT>/judge_<model>/{cards,overlaps,hw_profile}.json, keys/<model>.json,
                                  #    prompts/judge_<model>.md
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prompts"))
from gen import PROGRAMS, REPO, ROOT  # noqa: E402


def main():
    model = sys.argv[1]
    box = f"{ROOT}/judge_{model}"
    if os.path.exists(box):
        raise SystemExit(f"{box} exists")
    os.makedirs(f"{box}/out")
    os.makedirs(f"{HERE}/keys", exist_ok=True)
    subprocess.run(
        [sys.executable, f"{HERE}/pool.py", model, box, f"{HERE}/keys/{model}.json"],
        check=True,
    )
    shutil.copy(f"{HERE}/../hw_profile.json", f"{box}/hw_profile.json")
    for f in ("cards.json", "overlaps.json", "hw_profile.json"):
        os.chmod(f"{box}/{f}", 0o444)
    name, program = PROGRAMS[model]
    text = (
        open(f"{HERE}/JUDGE_TEMPLATE.md")
        .read()
        .replace("{MODEL_NAME}", name)
        .replace("{PROGRAM}", program)
        .replace("{BOX}", box)
        .replace("{REPO}", REPO)
    )
    assert "{" + "BOX}" not in text
    with open(f"{HERE}/../prompts/judge_{model}.md", "w") as f:
        f.write(text)
    print(f"wrote prompts/judge_{model}.md")


if __name__ == "__main__":
    main()

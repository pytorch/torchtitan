You are working on the **TorchTitan** training framework at the repository root.
Use your full toolset (Read, Grep, Glob, Bash, Edit, Write) as needed to complete
the task end-to-end.

**Fixed evaluation config (do not change the mesh).** Pipeline parallel = 4,
expert parallel = 2, data-parallel shard = auto (`-1`, fills the mesh; EP is carved
from the data axis, so on {{NGPU}} GPUs the data axis is 2 and EP=2 shards experts
across it); sequence length 2048; global batch size 1024; BF16 precision. Model
under test: `MODULE={{MODULE}}`, `CONFIG={{CONFIG}}` (an MoE model).

**Launch training with the provided script (treat it as read-only):**
```
bash {{TRAIN_SH}}      # env knobs: STEPS, MODULE, CONFIG, NGPU, PP, EP, DP
```
It wraps TorchTitan's `./run_train.sh` with the fixed config above. For a no-GPU
config check you may use `COMM_MODE=fake_backend PP=1 EP=1 DP=1`.

**Write every task artifact under `{{WORKSPACE}}/{{LABEL}}/`** exactly as the task
specifies — the grader checks the artifact, not the path you took to produce it.

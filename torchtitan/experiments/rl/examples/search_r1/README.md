# Search-R1: multi-turn retrieval-augmented GRPO

A [Search-R1](https://github.com/PeterGriffinJin/Search-R1) example in torchtitan's
RL experiment. The model is given a `search` tool; it thinks natively
(`enable_thinking=True`), calls `search` (a standard tool call) when it needs facts,
gets the retrieved passages back as a `tool` message, and replies with a final answer.
Reward is exact-match (EM) against gold answers (optionally plus a retrieval bonus).

It is a multi-turn, tool-using RL example: each assistant turn ends at the renderer's
role boundary, the env answers a `search` tool call with a `tool`-role message, and the
rollout continues until the model stops calling tools or the turn budget is hit. It
runs entirely on the framework's multi-turn rollouter (`rollout/rollouter.py`) and
continuous-batching generator — the only example-specific pieces are this folder plus
its config.

## Files
- `data.py` — `SearchR1Dataset` / `SearchR1Sample`: streams the NQ/HotpotQA parquet,
  downloaded from the HF dataset `PeterJinGo/nq_hotpotqa_train` (no preprocessing).
- `env.py` — `SearchR1Env(MessageEnv)`: defines the `search` `ToolSpec`, reads the
  renderer-parsed `tool_calls`, runs retrieval, and returns the passages as a `tool`
  message. The per-rollout turn budget is enforced by `TokenEnv.max_num_turns`.
- `rubric.py` — `RewardExactMatch`. **Default = pure-EM 0/1** on the final answer
  (correct → 1.0, else 0). Set `retrieval_score` > 0 to give partial credit when a
  search surfaced the gold answer (puts search on the gradient, anti closed-book).
- `rollouter.py` — wires datasets + env + rubric into a `Rollouter.Config`.

## Prerequisites

Install the example's extra Python dependencies (not in core torchtitan):

```bash
pip install -r torchtitan/experiments/rl/examples/search_r1/requirements.txt
```

### 1. Data
Nothing to prepare — the NQ/HotpotQA parquet is pulled straight from the HF dataset
[`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)
on first use (train + NQ-test splits). To use a local copy instead, set the dataset
config's `data_path` to a parquet with `question` / `golden_answers` columns.

### 2. Local dense retrieval server
Start the dense retriever (e5 index over wiki-18) listening on
`http://127.0.0.1:8000/retrieve` **before** training, pinned to spare GPU(s) so it does
not clash with the RL GPUs:

```bash
python <search-r1>/local_dense_retriever/retrieval_server.py \
  --index_path $INDEX_PATH/e5_Flat.index \
  --corpus_path $CORPUS_PATH/wiki-18.jsonl \
  --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
```

Override `message_env.search_url` / `message_env.topk` in the config if needed.

## Run

```bash
# eval run (Qwen3-1.7B), W&B on
python torchtitan/experiments/rl/train.py \
  --module torchtitan.experiments.rl.examples.search_r1 \
  --config rl_grpo_qwen3_1_7b_search_r1
```

Watch `validation_reward/_mean` (NQ test EM) trend up.

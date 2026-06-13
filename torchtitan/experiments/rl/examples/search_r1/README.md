# Search-R1: multi-turn retrieval-augmented GRPO

A minimal reproduction of [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
in torchtitan's RL experiment, modeled on slime's `examples/search-r1`. The model
reasons in `<think>`, issues `<search>query</search>` calls (answered with retrieved
`<information>...</information>`), and finishes with `<answer>...</answer>`. Reward is
exact-match (EM) against gold answers (optionally plus a graded format/retrieval bonus).

It is a multi-turn, tool-using RL example: each `<search>` ends an assistant turn, the
env returns the retrieved passages as the next user message, and the rollout continues
until the model answers or the turn budget is hit. It runs entirely on the framework's
multi-turn rollouter (`rollout/rollouter.py`) and continuous-batching generator — the
only example-specific pieces are this folder plus a config that sets the slime recipe.

## Files
- `data.py` — `SearchR1Dataset` / `SearchR1Example`: reads the Search-R1 NQ/HotpotQA parquet.
- `env.py` — `SearchR1Env(MessageEnv)`: text-tag `<search>`/`<answer>` protocol, injects
  `<information>`, and contains the local dense-retrieval client + the per-rollout
  `max_assistant_turns` budget.
- `rubric.py` — `RewardSearchR1` (slime/Search-R1 `compute_score_em`). **Default =
  slime's pure-EM 0/1** (correct answer → 1.0, else 0). Opt into the fine-grained
  graded reward by setting the sub-scores > 0
  (`structure_format_score=0.2, retrieval_score=0.1, final_format_score=0.1`): it adds
  a `<think>→<search>→<information>→<answer>` format state-machine + retrieval-correctness
  credit, so a bare correct answer (0.8) scores less than a searched one (1.0) and the
  policy can't reward-hack by skipping search. `RewardAnswerEM` is a metric-only (weight 0)
  pure-EM signal.
- `rollouter.py` — wires datasets + env + rubric into a `Rollouter.Config`.

## Prerequisites

Install the example's extra Python dependencies (not in core torchtitan):

```bash
pip install -r torchtitan/experiments/rl/examples/search_r1/requirements.txt
```

### 1. Data
Prepare the Search-R1 NQ/HotpotQA parquet (via Search-R1's
`scripts/data_process/qa_search_{train,test}_merge.py`). Point the config's
`train_dataset`/`validation_dataset` `data_path` at it, or set the
`SEARCH_R1_TRAIN_PARQUET` / `SEARCH_R1_TEST_PARQUET` env vars (used by `rollouter.py`).

### 2. Local dense retrieval server
Start the Search-R1 / slime dense retriever (e5 index over wiki-18) listening on
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

Watch `validation_reward/_mean` (EM, slime's `eval/nq_test` definition) trend up.

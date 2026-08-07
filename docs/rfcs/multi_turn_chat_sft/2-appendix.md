
## 9. Adding tool calls

Now layer tools onto the example. The data contract grows by one field; supervision policy doesn't change.

```python
messages_with_tools = [
    {"role": "user", "content": "weather in Paris?"},
    {"role": "assistant", "content": "I'll check.",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
    {"role": "tool", "content": "22 C, sunny", "tool_call_id": "c1"},
    {"role": "assistant", "content": "It's 22 C and sunny."},
]
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]

keep_loss = [False, True, False, True]
# user emits a question  -> not supervised
# assistant emits a tool call (model output) -> supervised
# tool emits a result (environment output) -> not supervised
# assistant emits the final answer (model output) -> supervised
```

**Wire-format diversity is a renderer concern, not a data-contract concern.** The same logical assistant tool call renders five different ways across models:

- Qwen3: `<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>`
- Llama 3.1: `<|python_tag|>...`
- DeepSeek-V3: `<｜tool▁calls▁begin｜>...<｜tool▁call▁end｜>`
- gpt-oss: `<|channel|>commentary to=functions.get_weather<|message|>{...}<|call|>`
- Nemotron 3: `<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>`

The algorithms in §5 (A1, A2, A4) all handle this because they don't parse model-specific markers — they let `apply_chat_template` (or a renderer) own the format and find spans by diffing or by direct attribution.

**One concrete bug surfaced by tool calls:** the message-payload-diff fallback inside A1 must clear `tool_calls=[]` alongside `content`. If we leave `tool_calls` in place when erasing the payload, the rendered tool-call XML appears in *both* the full render and the post-erasure render → diff window has length 0 → the entire assistant turn gets no supervision (verified by codex's Qwen3 probe in earlier review). The helper in §12 erases all assistant-generated fields, not just `content`.

## 10. Decisions

### Data contract

```python
@dataclass
class ChatSFTExample:
    messages: list[dict]                       # OpenAI-format with optional tool_calls
    tools: list[dict] | None = None
    keep_loss: list[bool] | None = None        # OPTIONAL per-message override
    chat_template_kwargs: dict | None = None
```

`messages + keep_loss` is the canonical form. If `keep_loss` is omitted, derive from a `train_on` policy enum (`ASSISTANT` / `LAST_ASSISTANT` / `ALL`). This decouples *what to supervise* from *how to render*; it expresses tool-call vs tool-response policy without per-template parsing; and it supports partial supervision ("supervise turn 4 but not turn 6") without flag explosion. **Decision: `messages + keep_loss` canonical, `train_on` derives a default.**

### Span-recovery algorithm

**Decision: A1 dispatcher (incremental render with prefix check, fall back to message-payload diff) for v1.** A2 (HF generation-mask) is opt-in for users with tagged templates. A3 (prompt subtraction) is the existing single-turn fast path. A4 (renderer-emitted) is the long-term gold standard when renderers grow `tokenize_message_parts`.

### Thinking-trace policy

**Decision: `AS_RENDERED` default + visible audit.** `PRESERVE_WITH_TEMPLATE` and `PER_ASSISTANT` are explicit opt-ins per §4.

### Where model-specific behavior lives

**Policy** (model-agnostic) lives in the data contract: `keep_loss[i] = role == "assistant"` works for every model. **Format** (per-model) lives in templates/renderers. This split satisfies "torchtitan stays clean — no per-model special-casing" without putting Jinja-engineering on every user.

NVIDIA's Nemotron stack illustrates the split: the *inference* template (HF) strips historical thinking; the *training* template (Megatron-LM `nemotron_nano_v2_custom_template`) is a separate, simpler Jinja with no cross-turn coupling. Different ownership for different jobs.

## 11. Evaluating the PR comment proposal

[Slim Frkha's reply on PR #2769](https://github.com/pytorch/torchtitan/pull/2769#issuecomment-4403992474):
> 1. Use `return_assistant_tokens_mask=True`. Most standard models already support it (cites trl#5471).
> 2. Train on all intermediate thinking traces. Dropping them is an inference optimization with negligible impact.
> 3. Users own the Jinja template. Torchtitan stays clean — no per-model special-casing.

The goal — keep torchtitan core clean — is correct. The proposal is mechanism A2 + B1 + `PRESERVE_WITH_TEMPLATE`. Three claims to evaluate.

### Claim 1 — "most standard models already support `{% generation %}`"

**Verdict: factually incorrect.** Live HF Hub probe of 8 major instruct families: 0 of 8 ship `{% generation %}` in the default `tokenizer_config.json` (Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B, DeepSeek-V3, Mistral-7B-Instruct-v0.3, gpt-oss-20b, Gemma-2-9b-it). Plus Nemotron 3 Nano (verified for `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, 12B-v2, 30B-A3B): no `{% generation %}` and the template strips historical thinking via `content.split('</think>')[1]`.

Qwen team explicitly **declined** [Hub PR #14](https://huggingface.co/Qwen/Qwen3-8B/discussions/14), citing GGUF/llama.cpp portability. The TRL issue [#5471](https://github.com/huggingface/trl/issues/5471) Slim cites is *the tracking issue listing models that don't have it*. TRL is reportedly migrating away from `assistant_only_loss` ([HF Forum: deprecation of assistant_only_loss](https://discuss.huggingface.co/t/deprecation-of-assistant-only-loss/175041)) precisely because text-based mask inference proved fragile.

Practical impact: following the recipe out-of-the-box returns `assistant_masks = [0] * len(input_ids)`, only a `warning_once`, and silently zeroed gradients (see §5 A2 failure path).

### Claim 2 — "negligible impact from dropping intermediate traces"

**Verdict: empirically unverified; framing elides a known trade-off.** Forking the template to keep `<think>` everywhere (Slim's preferred mode) creates **train/inference prefix-shape mismatch** — at training, the model sees prior `<think>` blocks; at inference the stock template strips them. Whether this hurts is empirical and unmeasured. NVIDIA accepts this mismatch in their production stack but doesn't publish numbers either way. The three policies in §4 cover the trade-off explicitly; calling one obviously correct without acknowledging the mismatch is incomplete.

### Claim 3 — "users own the Jinja template; torchtitan stays clean"

**Verdict: framing is misleading.** Complexity moves; it doesn't disappear. Concrete:

- 0/8 stock templates have `{% generation %}` (Claim 1). Users have to fork.
- TRL ships ~9 forked training templates per family (`trl/chat_templates/*_training.jinja`) precisely because users do not, in practice, write correct training templates.
- fairseq2 uses HF assistant_masks without forks AND without validation — silent zero-mask is a documented failure mode.
- Internal reference **does not use Jinja at all** for SFT training; the dataset producer carries `keep_loss: list[bool]` explicitly.
- Tinker and verifiers ship dedicated `Nemotron3Renderer` Python classes (`tinker_cookbook/renderers/nemotron3.py:4-52`, `verifiers/.../renderers/nemotron3.py`) — concrete evidence that "user owns Jinja" doesn't scale across model families even within OSS.

§10's split — policy in the data contract, format in templates/renderers — preserves the "torchtitan stays clean" goal without putting Jinja-engineering on every user.

### Bottom line

Slim's design (A2 + `PRESERVE_WITH_TEMPLATE`) is a **valid expert-user path** if the user has a verified tagged training template AND accepts the inference-shape mismatch. It is **not the right default** because (a) the "stock templates support it" premise is wrong, (b) the silent-zero-mask failure is severe, (c) "user owns Jinja" pushes complexity to a place that's harder to debug than a per-message bool. This RFC preserves the option as `PRESERVE_WITH_TEMPLATE` but does not centre it.

## 12. Guardrails / audit / appendix helpers

### Per-example validation

```python
def validate_render(input_ids, labels, messages):
    if len(input_ids) != len(labels):
        raise ValueError(f"length mismatch: ids={len(input_ids)} labels={len(labels)}")
    if all(label == -100 for label in labels):
        raise ValueError(
            f"all tokens masked out: 0 supervised. Likely cause: every message has "
            f"keep_loss=False, or A2 was used with a template lacking {{% generation %}}."
        )
```

### Dataset-level audit

```python
THINK_FIELDS = ("thinking", "reasoning", "reasoning_content")  # and list-style content parts
THINK_MARKERS = ("<think>", "</think>")

def audit_strip(tokenizer, messages, tools=None):
    """Detect dataset reasoning content that the chat template strips.

    Production code should also handle list-style content parts, e.g.
    {"type": "thinking", "text": "..."} — this snippet only covers the common
    string + structured-field cases.
    """
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    stripped_indices = []
    for i, msg in enumerate(messages[:-1]):  # skip terminal turn (kept by all templates)
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        has_reasoning = (
            ("<think>" in content and "</think>" in content)
            or any(msg.get(f) for f in THINK_FIELDS)
        )
        if has_reasoning and "<think>" in content and "<think>" not in rendered:
            stripped_indices.append(i)
    return stripped_indices
```

Audit output on the simple example with a Qwen3 model (12,450 conversations):

```text
Dataset audit (config: train_on=assistant, expansion=single, thinking_policy=as_rendered):

  Total examples:                                    12,450
  Assistant turns:                                   24,901
  Assistant turns with raw reasoning content:        18,326
  Stripped historical reasoning turns:                9,801   ← NOT supervised

  Span-recovery dispatcher used:
    Incremental + prefix check:    2,649 examples
    Blanked-message diff:          9,801 examples (prefix property broke)

  9,801 turns containing reasoning are stripped by the chat template before
  tokenization. To supervise: thinking_policy=PRESERVE_WITH_TEMPLATE OR
  expansion=PER_ASSISTANT.
```

Why audit, not error: erroring on every Qwen3 multi-turn dataset breaks the most common user case. Power users can set `strict_audit=True` to convert detected strips to errors.

### Appendix helpers

#### Blanked-message diff (used by A1's fallback)

```python
def erase_message_payload_for_diff(message):
    """Erase model-generated content for the diff. CRITICAL: this includes structured
    fields, not just `content`, otherwise tool_calls survive the erasure and the
    diff window comes out empty for tool-call turns. (This bit RC v4 and was caught
    by codex's Qwen3 probe.)"""
    erased = dict(message)
    if erased.get("role") == "assistant":
        for field in ("content", "thinking", "reasoning", "reasoning_content"):
            if field in erased:
                erased[field] = ""
        erased["tool_calls"] = []
    else:
        erased["content"] = ""
    return erased


def first_last_token_diff(a, b):
    """First and last token-position differences. Returns (None, None) if identical."""
    n = min(len(a), len(b))
    start = next((i for i in range(n) if a[i] != b[i]), None)
    if start is None:
        return None, None
    end = len(a)
    j = 0
    while j < (len(a) - start) and j < (len(b) - start) and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        end -= 1
        j += 1
    return start, end


def recover_span_by_message_payload_diff(tokenizer, messages, message_index, full_render):
    """A1 fallback: find one message's span via message-payload-diff."""
    erased = list(messages)
    erased[message_index] = erase_message_payload_for_diff(messages[message_index])
    erased_render = tokenizer.apply_chat_template(erased, tokenize=True, add_generation_prompt=False)
    span_start, span_end = first_last_token_diff(full_render, erased_render)
    if span_start is None or span_start == span_end:
        raise ValueError(f"empty diff for message {message_index}: erasure did not remove rendered output")
    return span_start, span_end
```

#### Sentinel collision (production hardening)

The default `erase_message_payload_for_diff` leaves `content=""`. If a template emits identical structure for empty vs non-empty content (no diff window), production code should choose a sentinel string absent from the rendered output:

```python
def choose_sentinel(tokenizer, messages, tools):
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    for candidate in ("[[__SFT_BLANK_0__]]", "[[__SFT_BLANK_1__]]", "[[__SFT_BLANK_2__]]"):
        if candidate not in rendered:
            return candidate
    raise ValueError("could not find a sentinel absent from the rendered string")
```

#### `PER_ASSISTANT` + explicit `keep_loss`

If the user supplies explicit `keep_loss`, `PER_ASSISTANT` expansion targets only assistant turns with `keep_loss[i] == True`. The default `train_on=ASSISTANT` derivation expands all assistant turns.

## 13. Out of scope

- **Renderer Protocol extension** (`tokenize_message_parts(msg, tools=)`). A4 needs this. Future PR.
- **VLM/multimodal SFT.** TRL's `assistant_only_loss` doesn't work with VLMs; HF's `return_assistant_tokens_mask` had/has issues with `ProcessorMixin` ([transformers#36713](https://github.com/huggingface/transformers/issues/36713)). A4 generalizes if the renderer handles multimodal segments; future RFC.
- **`expansion=AUTO`.** On Qwen3/DeepSeek/Nemotron prefix breaks exactly when it matters → degenerates to `PER_ASSISTANT`. On non-thinking models prefix never breaks → degenerates to `SINGLE`. Adds complexity without buying behavior.
- **Empirical comparison of the three thinking policies.** This RFC identifies trade-offs without measurement. Future work: compare `AS_RENDERED` / `PER_ASSISTANT` / `PRESERVE_WITH_TEMPLATE` on downstream eval (MATH, MMLU-CoT) for a small thinking-model trained on a multi-turn reasoning dataset.
- **EOM/EOT-only fine-grained masking.** A4 with `tokenize_message_parts` enables this; not v1.
- **Code/file placement.** Where the SFT dataset class lives in torchtitan is a follow-up PR.

## 14. References — by mechanism family

Per-framework citations grouped by mechanism. Detailed line citations are in research notes under `discussions/28_SFT_chat_dataset/research/{opus,codex}/`.

### A1 — span recovery
- **NeMo-RL.** `nemo_rl/data/llm_message_utils.py:144-178, 515-540`; `nemo_rl/algorithms/sft.py:267-270`. Includes Nemotron 3 Nano SFT path (`examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml:1-8`) — silently fragile under the strip-aware Nemotron template.
- **prime-rl.** `src/prime_rl/utils/chat_template.py:110-150` (incremental-with-extension-check; raises `IncrementalTokenizationError`).
- **Verl.** `verl/utils/dataset/multiturn_sft_dataset.py:212-239, 423-454` (per-message tokenize + full-render alignment assertion; warns on Qwen Thinking).
- **Open-Instruct.** `open_instruct/dataset_transformation.py:1112-1173` (`mask_labels` with `add_generation_prompt=True` trick).
- **OpenRLHF.** `openrlhf/datasets/sft_dataset.py:94-135`.
- **Axolotl.** `src/axolotl/prompt_strategies/chat_template.py:648-763` (`find_turn` with `[[dummy_message]]` sentinel — same pattern as A1's message-payload-diff fallback).
- **Megatron-LM.** `megatron/core/tokenizers/text/libraries/sft_tokenizer.py:130-209`. Per-turn re-tokenize + `np.allclose` byte-equality assertion. Calls `apply_chat_template(chat_template=custom_template)` with the library-supplied training-only template (B2 sub-pattern).

### A2 — HF generation-mask
- **TRL.** `trl/trainer/sft_trainer.py:1166-1171, 1318-1325, 1404-1468, 485-494`. Forked templates at `trl/chat_templates/{qwen3,llama3,gemma,deepseekv3,...}_training.jinja`. DeepSeek-R1 prompt/completion split workaround at `data_utils.py:267-298` shows the issue isn't Qwen-only.
- **fairseq2.** `recipes/lm/sft/dataset.py:204-235`. No `if not any(mask): raise` — silent zero-mask risk.

### A3 — prompt subtraction (last-only)
- **SkyRL.** `skyrl/train/sft_trainer.py:94-145`.
- **ROLL.** `roll/pipeline/sft/sft_pipeline.py:62-74`.
- **AReaL.** `areal/dataset/{gsm8k,clevr_count_70k,...}.py`.

### A4 — renderer-emitted spans
- **Internal reference.** Per-message `keep_loss: list[bool]` data contract; per-model renderer hierarchy.
- **torchtune.** `torchtune/models/llama3/_tokenizer.py:300-322`; `torchtune/data/_messages.py:23-26, 905-935`.
- **Tinker.** `tinker_cookbook/renderers/base.py:1080-1100, 1519-1643`; `renderers/qwen3.py:88-161`; `renderers/nemotron3.py:4-52, 182-232, 299-410` (Nemotron-specific Python renderer with XML tools, low-effort, history truncation).
- **verifiers (renderer package).** `verifiers/packages/renderers/renderers/base.py:281-323, 433-452`; `nemotron3.py:1-13`.
- **LLaMA-Factory.** `src/llamafactory/data/template.py:407-472` (`Template` + `ReasoningTemplate`).
- **Slime (partial).** `slime/utils/mask_utils.py:55-125, 127-196`.

### C2 — branching
- **Tinker (Kimi K2 only).** `tinker_cookbook/renderers/kimi_k2.py:335-394`.

### HF transformers internals
- `transformers/tokenization_utils_base.py:3037-3060, 3088-3109` (`return_assistant_tokens_mask` builder).
- `transformers/utils/chat_template_utils.py:415-455, 492-495` (`AssistantTracker` Jinja extension; `warning_once` on missing `{% generation %}`).

### Live HF Hub probes
- Qwen3 stock template: `tokenizer_config.json:230` of `Qwen/Qwen3-0.6B`.
- Qwen3 declined `{% generation %}`: [Qwen/Qwen3-8B Hub PR #14](https://huggingface.co/Qwen/Qwen3-8B/discussions/14).
- Nemotron 3 Nano template (XML tool format): `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, 12B-v2, 30B-A3B model cards.
- gpt-oss stock template (dual close tokens): `chat_template.jinja:302-314` of `openai/gpt-oss-20b`.

### External
- [PR 2769 (torchtitan)](https://github.com/pytorch/torchtitan/pull/2769)
- [Slim's PR comment](https://github.com/pytorch/torchtitan/pull/2769#issuecomment-4403992474)
- [trl#5471](https://github.com/huggingface/trl/issues/5471)
- [transformers#30650](https://github.com/huggingface/transformers/pull/30650)
- [transformers#36713](https://github.com/huggingface/transformers/issues/36713)
- [HF Forum: deprecation of assistant_only_loss](https://discuss.huggingface.co/t/deprecation-of-assistant-only-loss/175041)

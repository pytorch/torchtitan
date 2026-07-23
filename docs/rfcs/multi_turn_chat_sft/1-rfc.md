# Multi-turn SFT chat datasets
## TL;DR

This RFC decides how torchtitan should turn a multi-turn `messages` list into `(input_ids, labels)` for SFT.

- **Data contract:** `messages + keep_loss`, where `keep_loss[i]` says whether message `i` should get cross-entropy loss.
- **V1 algorithm:** render with the model's stock chat template, recover per-message token spans by incremental rendering, and fall back to message-payload diff when the template rewrites history.
- **Long-term production grade** A Python renderer per model family. It can run as-rendered, preserve reasoning, or branch per assistant turn as an explicit policy choice.
- **Thinking policy:** default to `AS_RENDERED` plus a visible audit. If the audit shows stripped intermediate reasoning and you want that reasoning supervised, opt into `PRESERVE_WITH_TEMPLATE` or `PER_ASSISTANT`.

`return_assistant_tokens_mask=True` is a valid opt-in when a verified `{% generation %}` template is available. It should not be the default: 0 of 8 probed stock instruct templates had those markers.

## 1. Problem

Multi-turn SFT takes `messages` and produces `(input_ids, labels)` for cross-entropy. For modern chat models — Qwen3, DeepSeek-R1, Nemotron 3, gpt-oss — the chat template can rewrite history, strip reasoning, or use position-dependent close tokens. The default approach silently zeros gradients on real datasets.

This RFC picks a data contract, a span-recovery algorithm, and a thinking-trace policy. Where the code lives is a follow-up PR.

## 2. SFT vs RL in multi-turn preparation

```text
SFT:
  dataset row -> render the full known conversation -> build labels
  Labels choose which existing tokens get cross-entropy loss.
  Re-tokenization is free; the only invariant is labels[i] in {-100, input_ids[i]}.

RL:
  prompt -> policy SAMPLES next tokens -> environment/tool result arrives -> next turn
  Rollout code must preserve exactly what was sampled, because those tokens
  define logprobs, rewards, and later prompts. Re-rendering history that drops
  or rewrites sampled tokens makes PPO/GRPO importance ratios meaningless.
```

This is why SFT can choose among Jinja masks, span recovery, and renderer-emitted spans, while RL frameworks more often use Python renderers (e.g. `Qwen3Renderer.extend_prompt`) to preserve sampled-token continuity across turns. The rest of this doc focuses on SFT.

## 3. The simple example

```python
# A 4-message multi-turn chat with thinking traces. No tools yet.
messages = [
    {"role": "user",      "content": "What is 1+1?"},
    {"role": "assistant", "content": "<think>basic arithmetic</think> 2"},
    {"role": "user",      "content": "What is 4+4?"},
    {"role": "assistant", "content": "<think>basic arithmetic</think> 8"},
]
# Named chunks throughout this doc:
# [USER_1] [ASST_1] [USER_2] [ASST_2]
```

For SFT we want the model to learn what an assistant says, so:
```text
keep_loss = [False, True, False, True]
            user    asst   user    asst
```

**A subtle separation: label decision vs context decision.**

Saying "supervise ASST_1 and ASST_2" doesn't fully define the training input. Two questions are independent:

- **Label decision:** which token positions get cross-entropy loss?
- **Context decision:** which prior tokens are visible during the forward pass?

Concretely, when computing loss on ASST_2, does the forward pass see ASST_1's `<think>` or not? The answer depends on the chat template:

```text
AS_RENDERED  (Qwen3 / DeepSeek-R1 / Nemotron 3 / gpt-oss stock templates):
  input  = [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]
  loss   =    0              1                0              1
  ASST_2 forward pass does NOT see ASST_1's <think>. This matches the stock-rendered history context ASST_2 would see at inference time. Caveat: ASST_1 itself is supervised in its post-strip history shape, not in the fresh single-turn generation shape where it may start with a <think> opener.

PRESERVE_WITH_TEMPLATE  (TRL training fork; some Python renderer configs):
  input  = [USER_1] [ASST_1_WITH_THINK] [USER_2] [ASST_2_WITH_THINK]
  loss   =    0             1              0              1
  ASST_2 forward pass DOES see ASST_1's <think>. More CoT signal, but the
  prefix shape diverges from what the stock inference template renders.

PER_ASSISTANT  (branch into one row per assistant turn):
  row A: [USER_1] [ASST_1_WITH_THINK]                         loss=0 1
  row B: [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]   loss=0 0 0 1
  Each supervised assistant turn is terminal in its row, so its <think>
  is preserved and the forward-pass prefix matches stock inference.
```

§4 picks among these three. The rest of the doc (§5-§7) explains *how* libraries actually build `(input_ids, labels)` once a policy is chosen.

## 4. The thinking-trace choice

Three policies. The policy is the user's first decision; the mechanism (§5) is "which span-recovery algorithm matches my chosen policy?"

### `AS_RENDERED` (recommended default)

Trust the chat template; supervise whatever survives; audit visibly so the user knows what was lost.

- **Pros:** matches the stock-rendered history/context for later turns; one row per conversation; no template forking.
- **Cons:** historical reasoning is unsupervised under templates that strip it (Qwen3, DeepSeek-R1, Nemotron 3).
- **Recommendation:** default. Pair with the dataset audit in §12 so the user sees the count of stripped turns.

### `PRESERVE_WITH_TEMPLATE`

User supplies a forked Jinja template (or chooses a renderer config) that re-emits `<think>` for every assistant turn. TRL's `qwen3_training.jinja` is the canonical example: it removes the `loop.index0 > ns.last_query_index` conditional and always renders the thinking block.

- **Pros:** one row per conversation; both `<think>` blocks supervised.
- **Cons:** train/inference prefix-shape mismatch — at training the model sees prior `<think>` blocks; at inference the production server's stock template strips them. Whether this hurts is empirical and unmeasured. NVIDIA's Nemotron stack accepts this mismatch (their Megatron training template doesn't strip; their HF inference template does); they don't publish numbers either way.

### `PER_ASSISTANT`

Branch the conversation so every assistant turn appears as the *terminal* turn of some row. Templates that strip on history will keep `<think>` for the terminal turn of each row.

- **Pros:** every supervised turn matches inference distribution; both `<think>` blocks get gradient (each as a terminal turn in different rows).
- **Cons:** ~K× sample multiplication; mixing weights must compensate (count supervised tokens, not raw rows; see §12).

**Decision: default `AS_RENDERED` + audit. The other two are explicit opt-ins.**


## 5. Next sessions quickmap

### Axis A — how libraries build the loss mask
- **A1 — Span recovery:** render the full conversation; recover per-message token spans by re-rendering prefixes (with a fallback for templates that rewrite history).
- **A2 — HF generation-mask:** Jinja's `{% generation %}` tag plus `apply_chat_template(return_assistant_tokens_mask=True)` returns a per-token mask directly.
- **A3 — Final-only prompt subtraction:** tokenize prompt vs full conversation; supervise only the suffix.
- **A4 — Renderer-emitted spans:** a Python class returns tokens with per-message attribution as it writes them.

###  Axis B — who owns the chat format?
- **B1 — User edits the model's stock chat template** (manual Jinja editing).
- **B2 — Library ships a training template** (one or more Jinja files maintained by the framework).
- **B3 — Library bypasses Jinja entirely** (Python renderer/tokenizer code IS the format).

###  C - one row or branched rows?
- **C1 SINGLE:** one conversation → one row.
- **C2 PER_ASSISTANT:** one conversation → one row per supervised assistant turn.

###  Decision:
**V1 default path = A1 + B1 + C1 + AS_RENDERED.** Render with the model's stock HF chat template, recover assistant spans and keep one row per conversation.

**Long-term renderer path = A4 + B3 + C1.** A Python renderer emits tokens and token ownership together. It can run as-rendered, preserve reasoning, or branch per assistant turn as an explicit policy choice.

## 5.1 Axis A — how libraries build the loss mask

Output blocks below all run on the simple example with a Qwen3 stock template.

### A1. Span recovery

- **A1 — Span recovery:** render the full conversation; recover per-message token spans by re-rendering prefixes (with a fallback for templates that rewrite history).
- **A2 — HF generation-mask:** Jinja's `{% generation %}` tag plus `apply_chat_template(return_assistant_tokens_mask=True)` returns a per-token mask directly.
- **A3 — Final-only prompt subtraction:** tokenize prompt vs full conversation; supervise only the suffix.
- **A4 — Renderer-emitted spans:** a Python class returns tokens with per-message attribution as it writes them.

Used by **NeMo-RL, prime-rl, Verl, Open-Instruct, OpenRLHF, Axolotl, Megatron-LM**. Render the full conversation, then walk message-by-message to find which tokens came from which message.

```python
def a1_span_recovery_build_loss_mask(messages, tokenizer):
    """A1 pattern: render full conversation; recover per-message spans."""

    # Step 1: render the WHOLE conversation. All `labels` indices we set must
    # point INTO this array.
    full_render_token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    labels = [-100] * len(full_render_token_ids)

    # Step 2: walk message-by-message, asking "which slice of full_render_token_ids
    # came from this message?" by re-rendering increasingly long prefixes.
    rendered_through_previous = []  # tokens from messages[:i] (empty before the loop)

    for message_index, message in enumerate(messages):
        rendered_through_current = tokenizer.apply_chat_template(
            messages[: message_index + 1], tokenize=True, add_generation_prompt=False,
        )

        # The "prefix check": did rendering messages[:i+1] just APPEND tokens to
        # what messages[:i] produced? If yes, message i's tokens are exactly the
        # suffix that got added. If no, the template rewrote earlier tokens
        # (e.g., Qwen3 strips ASST_1's <think> once ASST_2 arrives).
        prefix_property_holds = (
            len(rendered_through_current) >= len(rendered_through_previous)
            and rendered_through_current[: len(rendered_through_previous)] == rendered_through_previous
            and full_render_token_ids[: len(rendered_through_current)] == rendered_through_current
        )

        if prefix_property_holds:
            span_start = len(rendered_through_previous)
            span_end = len(rendered_through_current)
        else:
            # Fallback: message-payload diff. Render the conversation twice — once
            # normally, once with this message's payload erased — and the differing
            # range is this message's span. See §12 appendix for the helper.
            span_start, span_end = recover_span_by_message_payload_diff(
                tokenizer, messages, message_index, full_render_token_ids,
            )

        # If this is an assistant message, copy its tokens INTO labels (compute loss).
        if message["role"] == "assistant":
            labels[span_start:span_end] = full_render_token_ids[span_start:span_end]

        rendered_through_previous = rendered_through_current

    return full_render_token_ids, labels
```

**Output on the simple example (Qwen3 stock template):**
```text
input  = [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0              1                0              1

ASST_1's <think> was stripped by the template before tokenization.
No mask can supervise tokens that aren't in input_ids.
```

The fallback (message-payload diff) handles cases where the template rewrites history on later messages; details in §12.

### A2. HF generation-mask

Used by **TRL, fairseq2, verifiers (SFT script, via TRL)**. HuggingFace transformers has a Jinja tag `{% generation %}...{% endgeneration %}`. If a chat template wraps assistant content in that tag, calling `apply_chat_template(return_assistant_tokens_mask=True)` returns a per-token 0/1 mask.

```python
def a2_hf_generation_mask_build_loss_mask(messages, tokenizer):
    """A2 pattern: HF Jinja {% generation %} block emits the assistant mask directly."""

    # If the model's stock template doesn't have {% generation %}, swap in a
    # training-only template that does. TRL ships ~9 of these per family.
    if "{% generation %}" not in tokenizer.chat_template:
        tokenizer.chat_template = library_supplied_training_template_for(tokenizer)

    out = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
    )
    input_ids = out["input_ids"]
    assistant_mask = out["assistant_masks"]   # NOTE plural; the kwarg is singular

    if not any(assistant_mask):
        # If no assistant tokens were marked, the template lacks {% generation %}.
        # Cross-entropy with all -100 silently zeros the gradient. TRL guards;
        # fairseq2 does not.
        raise ValueError("template likely lacks {% generation %}")

    labels = [tok if keep else -100 for tok, keep in zip(input_ids, assistant_mask)]
    return input_ids, labels
```

**Output on the simple example (success path, with TRL's `qwen3_training.jinja`):**
```text
input  = [USER_1] [ASST_1_WITH_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0             1              0              1

The training template re-emits <think> for both assistant turns; {% generation %}
markers wrap the assistant body. One row, both <think> blocks supervised. But
training-input shape diverges from stock-inference shape.
```

**Output on the simple example (failure path, stock Qwen3 template):**
```text
assistant_mask = [0, 0, 0, ..., 0]   ← template lacks {% generation %}
outcome        = `if not any(mask): raise` in TRL-style robust code;
                 silent zero-loss gradient in weak code (fairseq2 has no check).
```

This failure path is real for 0/8 stock instruct templates verified live on the HF Hub (Llama-3.x, Qwen2.5, Qwen3, DeepSeek-V3, Mistral, gpt-oss, Gemma) plus Nemotron 3 Nano/Super. Qwen team explicitly declined [Hub PR #14](https://huggingface.co/Qwen/Qwen3-8B/discussions/14) to add the markers.

### A3. Final-only prompt subtraction

Used by **SkyRL, ROLL, AReaL**. Tokenize `messages[:-1]` with `add_generation_prompt=True` to get the prompt; tokenize the full conversation; mask the prompt and supervise the rest.

```python
def a3_final_only_build_loss_mask(messages, tokenizer):
    """A3 pattern: only supervise the final assistant message."""

    prompt_token_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
    )
    full_render_token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    prompt_length = len(prompt_token_ids)
    labels = [-100] * prompt_length + list(full_render_token_ids[prompt_length:])
    return full_render_token_ids, labels
```

**Output on the simple example:**
```text
input  = [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0              0                0              1
                  ^
                  ASST_1 gets NO loss. A3 cannot train multi-assistant SFT.
```

Useful for one-shot prompt/completion datasets. Not enough when you want loss on every assistant turn.

### A4. Renderer-emitted spans

Used by **torchtune, Tinker, LLaMA-Factory, Slime (partial), verifiers (renderer package), internal reference**. Instead of calling `apply_chat_template` and recovering spans afterward, a Python class encodes the chat format directly and returns `(tokens, owning_message)` pairs as it writes.

```python
def a4_renderer_emitted_build_loss_mask(messages, renderer):
    """A4 pattern: renderer writes tokens and reports which message owns each."""

    full_render_token_ids = []
    labels = []

    for message in messages:
        # Renderer is a Python class that knows the chat format. For Qwen3 it
        # emits <|im_start|>...<|im_end|>; for gpt-oss it emits channels;
        # for Nemotron 3 it emits <tool_call><function=...>; etc.
        # Returns three lists per message:
        #   header_ids: role marker tokens (NOT generated by the model)
        #   body_ids:   content tokens (FROM the model; usually what we supervise)
        #   end_ids:    end-of-turn tokens (NOT generated by the model)
        parts = renderer.tokenize_message_parts(message)
        all_message_tokens = parts["header_ids"] + parts["body_ids"] + parts["end_ids"]

        full_render_token_ids.extend(all_message_tokens)
        if message["role"] == "assistant":
            labels.extend(all_message_tokens)
        else:
            labels.extend([-100] * len(all_message_tokens))

    return full_render_token_ids, labels
```

A renderer is *just the chat template written in Python*, with token attribution returned alongside the tokens. It is not doing inference and should not parse rendered strings with regex.

**Output on the simple example (Qwen3Renderer with `strip_thinking_from_history=False`):**
```text
input  = [USER_1] [ASST_1_WITH_THINK] [USER_2] [ASST_2_WITH_THINK]
owner  =    0             1              2              3
loss   =    0             1              0              1

Both <think> blocks supervised in one row. Tradeoff: training input shape may
not match stock inference (renderer chose to preserve; stock template strips).
The same renderer with strip_thinking_from_history=True would match A1's
inference-shape output above.
```
### Axis A recap

```text
A1 span recovery       — works with any HF template; pays O(n_turns) extra renders.
A2 HF generation-mask  — fast when usable; 0/8 stock templates ship {% generation %}.
A3 final-only          — simplest; only trains the last assistant.
A4 renderer-emitted    — most precise; requires per-model Python code.
```

**Recommendation for torchtitan:**
v1 uses **A1**. Reason: works with stock HF templates;
**A4 long-term** when renderers expose `tokenize_message_parts` (a future renderer-protocol PR).

## 6. Axis B — who owns the chat format?

The four mechanisms differ on *who finds spans*. There's a separate question: *who supplies the chat template (or the Python renderer) that controls the rendering?*

- **B1 — User edits the model's stock chat template** (manual Jinja editing).
- **B2 — Library ships a training template** (one or more Jinja files maintained by the framework).
- **B3 — Library bypasses Jinja entirely** (Python renderer/tokenizer code IS the format).

We frame this section by asking the practical user question: *"if I want to keep ASST_1's `<think>` in the rendered output, where do I make that change?"*

### B1. User edits the model's stock chat template

Examples: **NeMo-RL, fairseq2, Axolotl, OpenRLHF, prime-rl, Verl**.

```python
# Default: library calls the tokenizer's stock template. Whatever it strips, you
# inherit. To keep ASST_1's <think>, the user opens the model's
# tokenizer_config.json, removes the strip conditional, and points the tokenizer
# at the patched template:
tokenizer.chat_template = open("my_qwen3_no_strip.jinja").read()
input_ids, labels = a1_span_recovery_build_loss_mask(messages, tokenizer)
```

**Output on the simple example, default (no edit):**
```text
input  = [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0              1                0              1
```

**Output after user edit (template no longer strips):**
```text
input  = [USER_1] [ASST_1_WITH_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0             1              0              1
```

The user is responsible for getting the Jinja correct. Most users won't.

### B2. Library ships a training template

Examples: **TRL** (registry of forks per family); **Megatron-LM** (one minimal training template + alignment check).

Two sub-patterns under B2:

```python
# B2 sub-pattern: registry of family templates (TRL's approach).
# The library ships ~9 forked Jinja files at trl/chat_templates/qwen3_training.jinja,
# llama3_training.jinja, etc. It detects the model family and swaps:
if "{% generation %}" not in tokenizer.chat_template:
    tokenizer.chat_template = get_training_chat_template(tokenizer)
input_ids, labels = a2_hf_generation_mask_build_loss_mask(messages, tokenizer)
```

```python
# B2 sub-pattern: one minimal training template + invariant check (Megatron-LM).
# The library ships ONE simple Jinja string with no cross-turn coupling, plus a
# per-turn alignment assertion to guarantee correctness:
input_ids = tokenizer.apply_chat_template(
    messages, chat_template=NEMOTRON_NANO_V2_CUSTOM_TEMPLATE,
    tokenize=True, add_generation_prompt=False,
)
# Then: per-turn re-tokenize and assert np.allclose against the joint render
# (sft_tokenizer.py:201). One template, provable correctness.
```

**Output on the simple example (B2, either sub-pattern):**
```text
input  = [USER_1] [ASST_1_WITH_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0             1              0              1

The library-supplied training template doesn't strip <think>. Both assistant
turns supervised. Tradeoff (same as B1 with edit): training input shape diverges
from stock inference.
```

### B3. Library bypasses Jinja for SFT

Examples: **torchtune, Tinker, LLaMA-Factory, internal reference, verifiers (renderer package), Slime (partial)**.

The library ships Python code (a "renderer" or "tokenizer class") that emits tokens directly. There is no Jinja involvement in the SFT path. *That code may preserve historical thinking, strip it, branch it, or expose it as configuration* — the benefit isn't "B3 always preserves," it's that formatting and token attribution are explicit Python with tests.

```python
# B3: Python renderer. Format is code, not Jinja.
# Many B3 renderers expose policy as configuration:
renderer = Qwen3Renderer(strip_thinking_from_history=False)  # preserve mode
input_ids, labels = a4_renderer_emitted_build_loss_mask(messages, renderer)
```

**Output on the simple example (B3 with preserve config):**
```text
input  = [USER_1] [ASST_1_WITH_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0             1              0              1
```

**Output on the simple example (B3 with as-rendered config):**
```text
input  = [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0              1                0              1
```

Same renderer code, different config flag. Per torchtune's source: `torchtune/models/llama3/_tokenizer.py` implements `tokenize_messages` directly; `torchtune/data/_messages.py:905-935` controls supervision via `train_on_all` / `train_on_assistant` / `train_on_last`. No HF Jinja for the SFT path.

### Axis B recap

```text
B1 user-edits-Jinja      — flexible but expects every user to be a Jinja expert.
B2 library-ships-Jinja   — TRL maintains ~9 forks; Megatron-LM ships one + check.
B3 bypass-Jinja          — Python renderer; format is code with tests.
```

**Recommendation:** keep policy model-agnostic via `keep_loss`; let format live in stock Jinja, training Jinja, or Python renderer; do not require every user to hand-author Jinja for correctness. Torchtitan can start in B1 (call tokenizer's stock template) and graduate to B3 (renderer registry from PR2) when renderers expose per-message tokenization.

## 7. Axis C — one row or branched rows?

Quick map:

- **C1 SINGLE:** one conversation → one row.
- **C2 PER_ASSISTANT:** one conversation → one row per supervised assistant turn.

```python
# C1: one row, both assistant turns share the same input_ids.
rows = [{
    "messages": [USER_1, ASST_1, USER_2, ASST_2],
    "keep_loss": [False, True, False, True],
}]
```

**Output on the simple example (C1 + AS_RENDERED):**
```text
input  = [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]
loss   =    0              1                0              1
```

```python
# C2: branch so each supervised assistant turn is terminal in its own row.
def expand_per_assistant(messages, keep_loss):
    rows = []
    for i in range(len(messages)):
        if keep_loss[i] and messages[i]["role"] == "assistant":
            sub_keep = [False] * (i + 1)
            sub_keep[i] = True
            rows.append({"messages": messages[:i + 1], "keep_loss": sub_keep})
    return rows

rows = expand_per_assistant(messages, [False, True, False, True])
# Returns two rows.
```

**Output on the simple example (C2 + AS_RENDERED):**
```text
row A: input = [USER_1] [ASST_1_WITH_THINK]                     loss = 0  1
       (ASST_1 is terminal here; <think> kept by the template.)

row B: input = [USER_1] [ASST_1_WITHOUT_THINK] [USER_2] [ASST_2_WITH_THINK]
       loss  =    0              0                0              1
       (ASST_2 is terminal; <think> kept. ASST_1's history-shape matches
       what the inference server will render when serving turn 2.)
```

Each row's prefix matches what the inference server would see when the model produces that specific assistant turn. Cost: ~K rows per K-assistant-turn conversation.

### Axis C recap

```text
C1 single sample      — what most SFT libraries do; avoids row explosion.
C2 branched samples   — preserves terminal-turn thinking; ~K× row count.
```

**Recommendation:** default **C1** because that's what most production SFT libraries do and it avoids row explosion.

## 8. Recommended paths

**V1 default path = A1 + B1 + C1 + AS_RENDERED.** Render with the model's stock HF chat template, recover assistant spans and keep one row per conversation.

Why this is the v1 recommendation:
- No per-model code is required; the tokenizer already ships the chat template.
- `keep_loss` stays model-agnostic: the dataset says what to supervise, while the template says how to render it.
- Later assistant turns see stock-rendered history. Caveat: non-terminal assistant turns are supervised in their post-strip history shape, not in their fresh single-turn generation shape.
- This is the common OSS shape. Examples: NeMo-RL, prime-rl, Verl, Open-Instruct, OpenRLHF, Axolotl. Close variants: Megatron-LM is A1+B2+C1; TRL is A2+B2+C1.

What this gives up:
- Qwen3, DeepSeek-R1, Nemotron 3, and gpt-oss strip historical thinking, so those tokens never enter `input_ids` and get no loss. The audit reports this.
- Span recovery costs extra `apply_chat_template` calls.
- The message-payload-diff fallback must erase all rendered payload fields, including `tool_calls`, and must use a sentinel that cannot collide with data.
- Stock-template tool-format bugs remain stock-template bugs.

**Long-term renderer path = A4 + B3 + C1.** A Python renderer emits tokens and token ownership together. It can run as-rendered, preserve reasoning, or branch per assistant turn as an explicit policy choice.

Examples: torchtune, Tinker (except its Kimi K2 path which is C2), LLaMA-Factory, internal reference, verifiers (renderer package), slime.

Why this is the long-term target:
- Token ownership is produced directly; no span recovery heuristic is needed.
- Tool-call wire format, thinking policy, and close-token behavior are explicit code with parity tests.
- It matches the RL renderer direction, but SFT can adopt it only when renderers expose per-message tokenization.

What this costs:
- Per-model renderer work and tests.
- A renderer protocol extension (`tokenize_message_parts` or equivalent).
- Not necessary for the first SFT PR.

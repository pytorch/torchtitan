# Multi-turn SFT chat datasets — without footguns

> RC v6 by opus (claude-opus-4-7). Companion to `MANIFEST.md`. Public-facing.

## TL;DR

This RFC decides how torchtitan should turn a multi-turn `messages` list into `(input_ids, labels)` for SFT.

- **Data contract:** `messages + optional keep_loss`, where `keep_loss[i]` says whether message `i` should get cross-entropy loss.
- **V1 algorithm:** render with the model's stock chat template, recover per-message token spans by incremental rendering, and fall back to message-payload diff when the template rewrites history.
- **Thinking policy:** default to `AS_RENDERED` plus a visible audit. If the audit shows stripped intermediate reasoning and you want that reasoning supervised, opt into `PRESERVE_WITH_TEMPLATE` or `PER_ASSISTANT`.

`return_assistant_tokens_mask=True` is a valid opt-in when a verified `{% generation %}` template is available. It should not be the default: 0 of 8 probed stock instruct templates had those markers.

## 1. Problem

Multi-turn SFT takes `messages` and produces `(input_ids, labels)` for cross-entropy. For modern chat models — Qwen3, DeepSeek-R1, Nemotron 3, gpt-oss — the chat template can rewrite history, strip reasoning, or use position-dependent close tokens. The default approach silently zeros gradients on real datasets.

This RFC picks a data contract, a span-recovery algorithm, and a thinking-trace policy. Where the code lives is a follow-up PR.

**Answer preview:** dataset rows are `messages + optional keep_loss`. Default policy supervises every assistant turn under a single rendered conversation, with a visible audit when the chat template strips intermediate reasoning. Two opt-in alternatives — `PRESERVE_WITH_TEMPLATE` and `PER_ASSISTANT` — handle reasoning-heavy datasets where the default's lost CoT supervision matters.

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

### Choosing

- Conversations short or no `<think>` in dataset → `AS_RENDERED` is a no-op.
- Long conversations with rich `<think>` in every turn → `PER_ASSISTANT`.
- Researcher with empirically validated forked template → `PRESERVE_WITH_TEMPLATE`.

**Decision: default `AS_RENDERED` + audit. The other two are explicit opt-ins.**

## 5. Axis A — how libraries build the loss mask

Quick map (each detailed below):

- **A1 — Span recovery:** render the full conversation; recover per-message token spans by re-rendering prefixes (with a fallback for templates that rewrite history).
- **A2 — HF generation-mask:** Jinja's `{% generation %}` tag plus `apply_chat_template(return_assistant_tokens_mask=True)` returns a per-token mask directly.
- **A3 — Final-only prompt subtraction:** tokenize prompt vs full conversation; supervise only the suffix.
- **A4 — Renderer-emitted spans:** a Python class returns tokens with per-message attribution as it writes them.

Output blocks below all run on the simple example with a Qwen3 stock template.

### A1. Span recovery

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

**Recommendation for torchtitan:** v1 uses **A1**. Reason: works with stock HF templates today; correctness via dispatcher (incremental → message-payload-diff fallback). **A2 opt-in** when a verified tagged template exists (TRL's path). **A3 only** for one-shot prompt/completion datasets. **A4 long-term** when renderers expose `tokenize_message_parts` (a future renderer-protocol PR).

## 6. Axis B — who owns the chat format?

The four mechanisms differ on *who finds spans*. There's a separate question: *who supplies the chat template (or the Python renderer) that controls the rendering?*

Quick map:

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

**Recommendation:** default **C1** because that's what most production SFT libraries do and it avoids row explosion. Offer **C2 (`PER_ASSISTANT`)** when preserving terminal-turn thinking is more important than dataset size. Mixture weights after C2 expansion should be **by supervised tokens, not by row count** (a 4-turn → 4-row expansion has 4× the rows but the same supervised-token volume).

## 8. Recommended paths

**V1 default path = A1 + B1 + C1 + AS_RENDERED.** Render with the model's stock HF chat template, recover assistant spans with the incremental → message-payload-diff dispatcher, and keep one row per conversation.

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

**Long-term renderer path = A4 + B3 + C1/C2.** A Python renderer emits tokens and token ownership together. It can run as-rendered, preserve reasoning, or branch per assistant turn as an explicit policy choice.

Why this is the long-term target:
- Token ownership is produced directly; no span recovery heuristic is needed.
- Tool-call wire format, thinking policy, and close-token behavior are explicit code with parity tests.
- It matches the RL renderer direction, but SFT can adopt it only when renderers expose per-message tokenization.

What this costs:
- Per-model renderer work and tests.
- A renderer protocol extension (`tokenize_message_parts` or equivalent).
- Not necessary for the first SFT PR.

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

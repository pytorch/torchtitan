# BPE Boundary Research: Label Masking in SFT Libraries

## The Problem

BPE tokenizers don't have a clean split property:
`tokenize(A + B) != tokenize(A) + tokenize(B)`

When A ends with partial characters or tokens that merge with the start of B, the token
boundary shifts. This means you can't determine where the prompt ends in the full token
sequence simply by counting `len(tokenize(prompt))` — those tokens may differ from the
prefix of `tokenize(prompt + response)`.

---

## Library Survey

### 1. torchtune (`HuggingFaceModelTokenizer`)

**Source:** `torchtune/modules/transforms/tokenizers/_hf_tokenizer.py`

**Technique: Incremental prefix re-tokenization (diff approach)**

For each message in the conversation, torchtune re-tokenizes the entire conversation up to
and including that message, then takes the _delta_ (new tokens beyond what was already
tokenized). Each message's tokens are defined as `current_tokens[len(previous_tokens):]`.

```python
for i, message in enumerate(messages):
    current_messages = [
        {"role": m.role, "content": m.content[0]["content"]}
        for m in messages[: i + 1]
    ]
    rendered = self.render_template(current_messages, add_eos=False)
    current_tokens = self.base_tokenizer.encode(rendered, add_eos=False)
    delta = current_tokens[len(previous_tokens):]
    previous_tokens = current_tokens
    tokenized_messages.extend(delta)
    mask.extend([message.masked] * len(delta))
```

The per-message `masked` flag on `Message` objects drives masking. Prompt messages have
`masked=True`; assistant messages have `masked=False`.

**Why this works:** Each delta is defined relative to the already-tokenized prefix, so
the BPE merges at the boundary are always resolved by re-tokenizing the full prefix. The
delta tokens are exactly the tokens that appear in the full tokenization that weren't
in the prior prefix.

**Pros:**
- Correct by construction: boundary tokens always match the full sequence
- Works for any chat template without special template markup
- Handles multi-turn naturally (each turn gets its own mask delta)

**Cons:**
- O(N^2) tokenizations per sample (one per message)
- Requires tokenizing N growing prefixes per example
- More complex implementation

---

### 2. TRL (`SFTTrainer`)

**Source:** `trl/trainer/sft_trainer.py` ~line 1057-1108

**Two techniques, depending on mode:**

#### Mode A: Separate tokenization with prefix-length masking

For `prompt-completion` conversational datasets:
1. Tokenize the prompt alone with `apply_chat_template(..., add_generation_prompt=True, tokenize=True)` → `prompt_ids`
2. Tokenize the full conversation `apply_chat_template(prompt + completion, ..., tokenize=True)` → `prompt_completion_ids`
3. Set `completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))`
4. Warn (but don't fail) if `prompt_completion_ids[:len(prompt_ids)] != prompt_ids`

```python
if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
    logger.warning(
        "Mismatch between tokenized prompt and the start of tokenized "
        "prompt+completion. This may be due to unexpected tokenizer behavior..."
    )
completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
```

This is the approach torchtitan currently uses (encoding prompt and full text
separately, then using `len(prompt_tokens)` as the boundary).

**The BPE risk:** If `tokenize(prompt)` produces different tokens at the prompt/response
boundary than `tokenize(prompt + response)`, the `completion_mask` will mask one token
too many or too few. TRL acknowledges this with a warning but doesn't fix it.

#### Mode B: Chat template `{% generation %}` markers (`assistant_only_loss=True`)

For templates that support it, TRL passes `return_assistant_tokens_mask=True` to
`apply_chat_template`. This relies on the Jinja template having `{% generation %}` and
`{% endgeneration %}` blocks marking assistant content. The HF tokenizer then returns
an `assistant_masks` array alongside the tokens via character-level tracking in the
template render.

```python
prompt_completion_processed = processing_class.apply_chat_template(
    prompt + completion,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=assistant_only_loss,
)
if "assistant_masks" in prompt_completion_processed:
    output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
```

**How `return_assistant_tokens_mask` works:** During template rendering, HF's Jinja
environment records the character positions of `{% generation %}` blocks. After
tokenization with `return_offsets_mapping=True`, it maps those character ranges to token
positions. This is a full offset_mapping approach — it avoids separate tokenization of
the prompt entirely.

**Pros (Mode B):** Correct by construction, single tokenization pass, no BPE boundary
problem.

**Cons (Mode B):** Requires the chat template to have `{% generation %}` markup. Most
production templates (Llama-3, Qwen, Mistral) do not have it. Template authors must opt in.

---

### 3. axolotl (`InstructionPromptTokenizingStrategy`)

**Source:** `axolotl/prompt_tokenizers.py`

**Technique: Separate tokenization and concatenation**

Axolotl tokenizes the prompt and response as completely separate strings, then
concatenates the token lists:

```python
tokenized_prompt = self._tokenize(user_prompt, add_eos_token=False)
if not self.train_on_inputs:
    user_prompt_len = len(tokenized_prompt["input_ids"])
    tokenized_prompt["labels"] = [IGNORE_INDEX] * user_prompt_len

tokenized_res_prompt = self._tokenize(
    response, strip_bos_token=True, add_eos_token=True
)
tokenized_prompt["input_ids"] += tokenized_res_prompt["input_ids"]
tokenized_prompt["labels"] += tokenized_res_prompt["input_ids"]
```

The BOS token is stripped from the response tokenization (`strip_bos_token=True`) so
the concatenation doesn't have a spurious BOS in the middle.

**The BPE boundary risk:** The tokens produced by this concatenation can differ from
`tokenize(prompt + response)` at the junction boundary. For most common templates, the
boundary falls between a special token (e.g., `<|eot_id|>` or `[/INST]`) and the start
of the assistant response. Special tokens suppress BPE merges across their boundaries, so
in practice the risk is low for well-designed templates.

**Pros:**
- Simplest implementation
- Single tokenization pass per part
- Works fine when template boundaries are special tokens

**Cons:**
- Incorrect if non-special-token text spans the boundary
- Requires stripping BOS from response (potential source of bugs)
- Token alignment between input_ids and labels handled manually

---

### 4. LLaMA-Factory (`template.encode_multiturn`)

**Source:** `src/llamafactory/data/template.py` + `processors/supervised.py`

**Technique: Per-element separate tokenization, concatenation by turn**

LLaMA-Factory builds conversations from `SLOTS` — small string fragments and special
token markers. Each slot is tokenized independently via `tokenizer.encode(elem, add_special_tokens=False)`.
A full turn's token list is the concatenation of all its slot token lists.

```python
def _convert_elements_to_ids(self, tokenizer, elements):
    token_ids = []
    for elem in elements:
        if isinstance(elem, str):
            token_ids += tokenizer.encode(elem, add_special_tokens=False)
        elif isinstance(elem, dict):
            token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
        elif isinstance(elem, set):
            if "bos_token" in elem:
                token_ids += [tokenizer.bos_token_id]
            ...
    return token_ids
```

`encode_multiturn` returns list of `(source_ids, target_ids)` pairs. The supervised
processor then labels `source_ids` as `IGNORE_INDEX` and `target_ids` as the actual
token ids.

**BPE boundary risk:** Same fundamental risk as axolotl. Each text fragment is tokenized
independently, so merges across slot boundaries don't happen. In LLaMA-Factory's case,
template slots are typically structured so that special token boundaries separate the
user text from assistant text, again reducing the practical risk.

**Pros:**
- Per-slot control over masking
- Works for any template defined as slots
- No need to re-tokenize growing prefixes

**Cons:**
- Template definitions must be implemented as slot lists (not Jinja templates)
- Correct only when slot boundaries coincide with BPE-safe points
- Doesn't generalize to arbitrary Jinja templates

---

## Summary Comparison

| Library | Technique | BPE-safe? | Template-agnostic? | Complexity |
|---------|-----------|-----------|-------------------|------------|
| torchtune | Incremental prefix re-tokenization (delta) | Yes | Yes | O(N^2) tokenizations |
| TRL (mode A) | Separate prompt/full tokenization, length prefix | No (warns) | Yes (Jinja) | 2 tokenizations |
| TRL (mode B) | `return_assistant_tokens_mask` + `{% generation %}` | Yes | No (needs marker) | 1 tokenization |
| axolotl | Separate part tokenization + concat | No (low risk) | Yes (custom prompters) | 1-2 tokenizations |
| LLaMA-Factory | Per-slot separate tokenization + concat | No (low risk) | No (slot templates) | N tokenizations |

---

## Current torchtitan Approach

`dataset.py` lines 81-97:

```python
full_text = self._tokenizer.apply_chat_template(messages)
prompt_text = self._tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)

full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=True)
prompt_len = len(self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False))

mask_len = min(prompt_len - 1, len(label_ids))
for i in range(mask_len):
    label_ids[i] = IGNORE_INDEX
```

This is the same as **TRL mode A**: encode prompt standalone, use its length as the mask
boundary in the full token sequence. It has the same BPE boundary risk.

---

## Recommendation

For torchtitan's specific situation (HuggingFaceTokenizer wrapping a HF fast tokenizer,
`apply_chat_template` returns a string, no `{% generation %}` markers assumed):

**Best approach: torchtune-style incremental prefix delta**

Tokenize growing prefixes. For each message, the tokens assigned to it are the delta
between the current prefix tokenization and the previous one. This is correct by
construction with no BPE boundary error.

Adapted to torchtitan, the change would be in `_tokenize_sample`:

```python
messages = self._sample_processor(sample)
full_text = self._tokenizer.apply_chat_template(messages)
full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=True)

# Build mask by re-tokenizing growing prefixes
label_ids = full_tokens[1:]
input_ids = full_tokens[:-1]

prev_len = 0
for i, message in enumerate(messages):
    prefix_text = self._tokenizer.apply_chat_template(
        messages[: i + 1],
        add_generation_prompt=(i < len(messages) - 1),
    )
    prefix_tokens = self._tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
    curr_len = len(prefix_tokens)
    if message["role"] != "assistant":
        mask_start = prev_len
        mask_end = min(curr_len - 1, len(label_ids))  # -1 for shift
        for j in range(mask_start, mask_end):
            label_ids[j] = IGNORE_INDEX
    prev_len = curr_len
```

**Why not TRL mode B (`{% generation %}`):** This requires all target chat templates to
have `{% generation %}` blocks. Llama-3.1, Qwen-2.5, and Mistral-0.3 do not. Requiring
template modifications is a significant user burden.

**Why not the simpler separate-tokenization approach (current):** In practice the current
approach is likely correct for most models because the prompt/response boundary in typical
chat templates falls immediately after a special token (`<|eot_id|>`, `<|im_end|>`,
`[/INST]`), which suppresses BPE merges. However it is technically incorrect and will
fail for templates that use plain-text delimiters like `\n\nAssistant:`.

**Tradeoff to consider:** The incremental approach requires O(N) tokenizations per sample
(where N = number of messages). For typical 2-turn (user+assistant) samples this is 2
tokenizations instead of 2, so there's no added cost. For long multi-turn conversations
this grows linearly. Given torchtitan's typical use case (single-turn instruction tuning),
this is acceptable.

**Simplest correct fix for single-turn use case:** A smaller, focused fix that avoids
the O(N) concern is to use `add_generation_prompt=True` when rendering the full
conversation with only the prompt messages (already done), but then instead of using the
prompt token count as a mask length, verify alignment by checking if the first
`prompt_len` tokens of `full_tokens` match the standalone `prompt_tokens`. If they
don't match, fall back to scanning for the EOS of the last non-assistant turn. This is
closer to what axolotl does with a safety check.

For torchtitan's current state, the most pragmatic recommendation is:
1. Keep the current two-tokenization approach (it's correct for all standard HF chat templates)
2. Add an assertion or warning if `full_tokens[:prompt_len] != prompt_tokens` (as TRL does)
3. Document this as a known limitation for non-standard templates

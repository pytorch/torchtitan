# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Zorplex RL - A synthetic benchmark for training LLMs on multi-step tool use.

Words map to hidden integer values via a seeded lookup table. The model must
use a LOOKUP[word] tool to discover values, then combine them with arithmetic.
"""

import operator
import random
import re
from dataclasses import dataclass, field

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


ZORPLEX_WORDS = [
    "apple",
    "banana",
    "cat",
    "dog",
    "elephant",
    "fish",
    "grape",
    "house",
    "ice",
    "jungle",
    "kite",
    "lemon",
    "moon",
    "night",
    "ocean",
    "piano",
    "queen",
    "river",
    "sun",
    "tree",
    "umbrella",
    "violet",
    "water",
    "xray",
    "yellow",
    "zebra",
]


def _make_table(seed: int = 42) -> dict[str, int]:
    """Create the zorplex lookup table."""
    rng = random.Random(seed)
    return {word: rng.randint(1, 100) for word in ZORPLEX_WORDS}


_SYSTEM_PROMPT = """You are a helpful assistant.

You have access to a LOOKUP tool. To find the zorplex value of a word, \
simply output LOOKUP[word] on its own line. The system will return the value \
to you. Do NOT write code - just output the tool call directly and wait for \
the result.

For problems requiring multiple lookups, call LOOKUP once, wait for the result, then call it again.

When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is zorplex('cat') + zorplex('dog')?
Assistant: LOOKUP[cat]
[Result: 42]
Assistant: LOOKUP[dog]
[Result: 17]
Assistant: 42 + 17 = 59. [ANSWER] 59"""

_TOOL_CALL_RE = re.compile(r"LOOKUP\[['\"]?(\w+)['\"]?\]", re.IGNORECASE)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Task:
    """A single task instance."""

    question: str
    correct_answer: int | str
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    """A parsed tool call from model output."""

    tool_name: str
    argument: str
    result: str | int | None = None


@dataclass
class Turn:
    """A single turn in the agentic loop."""

    generated_text: str
    tool_calls: list[ToolCall]
    tool_results: list[str]


@dataclass
class AgenticResult:
    """Result from agentic evaluation."""

    task: Task
    turns: list[Turn]
    final_text: str
    extracted_answer: int | str | None
    is_correct: bool

    @property
    def total_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)


# =============================================================================
# ZORPLEX SPEC
# =============================================================================


class ZorplexSpec:
    """Compositional zorplex task: multiple lookups + addition.

    Task: "What is zorplex('apple') + zorplex('banana')?"
    Tools: LOOKUP[apple] -> 82, LOOKUP[banana] -> 15
    Answer: 82 + 15 = 97
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)
        self._table = _make_table(seed)

    def generate_task(self) -> Task:
        words = self._rng.sample(ZORPLEX_WORDS, 2)
        values = [self._table[w] for w in words]
        answer = operator.add(values[0], values[1])
        question = f"What is zorplex('{words[0]}') + zorplex('{words[1]}')?"
        return Task(
            question=question,
            correct_answer=answer,
            metadata={"words": words, "values": values, "op": "+"},
        )

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        matches = _TOOL_CALL_RE.findall(text)
        return [ToolCall("LOOKUP", m.lower()) for m in matches]

    def execute_tool(self, tool_call: ToolCall) -> int | str:
        value = self._table.get(tool_call.argument.lower())
        return value if value is not None else "UNKNOWN"

    def extract_answer(self, text: str) -> int | None:
        # Check for [ANSWER] tag first
        answer_match = re.findall(r"\[ANSWER\]\s*(-?\d+)", text)
        if answer_match:
            return int(answer_match[-1])

        # Try to find explicit answer patterns
        patterns = [
            r"(?:the answer is|answer is|answer:)\s*(-?\d+)",
            r"=\s*(-?\d+)\.?\s*(?:The answer|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return int(matches[-1])

        # Fallback: last number in the text
        numbers = re.findall(r"-?\d+", text)
        return int(numbers[-1]) if numbers else None


# =============================================================================
# AGENTIC GENERATION
# =============================================================================


class _StopAtToolCall(StoppingCriteria):
    """Stop generation when a complete tool call is detected."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        new_tokens = input_ids[0, self.prompt_length :]
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return bool(_TOOL_CALL_RE.search(new_text))


def generate_with_tools(
    model,
    tokenizer,
    spec: ZorplexSpec,
    task: Task,
    device: str,
    max_turns: int = 5,
    max_tokens_per_turn: int = 150,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> AgenticResult:
    """Generate with iterative tool execution.

    Each turn:
    1. Generate text until a tool call is detected (or max tokens)
    2. Execute the tool call
    3. Inject result and continue
    4. Repeat until no more tool calls or max turns reached
    """
    messages = [
        {"role": "system", "content": spec.get_system_prompt()},
        {"role": "user", "content": task.question},
    ]

    turns: list[Turn] = []
    all_text = ""

    for turn_idx in range(max_turns):
        # Build prompt from conversation so far
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Add any previously generated text (continuing the assistant response)
        if all_text:
            text = text + all_text

        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]

        # Create stopping criteria that halts at tool calls
        stop_criteria = StoppingCriteriaList(
            [_StopAtToolCall(tokenizer, prompt_length)]
        )

        # Generate until tool call or max tokens
        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens_per_turn,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop_criteria,
                do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
            outputs = model.generate(**gen_kwargs)

        # Decode only the new tokens
        new_text = tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )

        # Find tool calls in the new text
        tool_calls = spec.parse_tool_calls(new_text)

        # Check if model emitted [ANSWER] tag — treat as final turn
        has_answer = bool(re.search(r"\[ANSWER\]\s*-?\d+", new_text, re.IGNORECASE))

        if not tool_calls or has_answer:
            # No tool calls - we're done
            all_text += new_text
            turns.append(
                Turn(
                    generated_text=new_text,
                    tool_calls=[],
                    tool_results=[],
                )
            )
            break

        # Execute tool calls and build result string
        tool_results = []
        for tc in tool_calls:
            result = spec.execute_tool(tc)
            tc.result = result
            tool_results.append(str(result))

        # Record this turn
        turns.append(
            Turn(
                generated_text=new_text,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
        )

        # Inject results and continue (format matches system prompt examples)
        result_injection = "\n[Result: " + ", ".join(tool_results) + "]\n"
        all_text += new_text + result_injection

    # Extract final answer from all generated text
    extracted = spec.extract_answer(all_text)
    is_correct = extracted == task.correct_answer

    return AgenticResult(
        task=task,
        turns=turns,
        final_text=all_text,
        extracted_answer=extracted,
        is_correct=is_correct,
    )

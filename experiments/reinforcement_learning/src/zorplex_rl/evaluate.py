"""
Agentic evaluation utilities for zorplex tasks.

Provides the core evaluation loop that generates with tool execution.
"""

import re
from dataclasses import dataclass

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .task_specs import TaskSpec, Task, ToolCall


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

    @property
    def full_trajectory(self) -> str:
        """Reconstruct the full conversation."""
        parts = []
        for turn in self.turns:
            parts.append(turn.generated_text)
            for tc, result in zip(turn.tool_calls, turn.tool_results):
                parts.append(f"\n[Tool result: {tc.tool_name}[{tc.argument}] = {result}]\n")
        return "".join(parts)


class StopAtToolCall(StoppingCriteria):
    """Stop generation when a complete tool call is detected."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        # Match tool calls: LOOKUP[word], GETKEY[word], FETCH[ZK_XXX]
        self.tool_pattern = re.compile(r"(LOOKUP|GETKEY|FETCH)\[[^\]]+\]", re.IGNORECASE)
        # Match final answer tag: [ANSWER] <number>
        self.answer_pattern = re.compile(r"\[ANSWER\]\s*-?\d+", re.IGNORECASE)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check newly generated tokens
        new_tokens = input_ids[0, self.prompt_length:]
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Stop if we find a complete tool call or an [ANSWER] tag
        return bool(self.tool_pattern.search(new_text) or self.answer_pattern.search(new_text))


def generate_with_tools(
    model,
    tokenizer,
    spec: TaskSpec,
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
        {"role": "system", "content": spec.get_system_prompt(with_hint=True)},
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
        stop_criteria = StoppingCriteriaList([
            StopAtToolCall(tokenizer, prompt_length)
        ])

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
            outputs[0][prompt_length:],
            skip_special_tokens=True
        )

        # Find tool calls in the new text
        tool_calls = spec.parse_tool_calls(new_text)

        # Check if model emitted [ANSWER] tag — treat as final turn
        has_answer = bool(re.search(r'\[ANSWER\]\s*-?\d+', new_text, re.IGNORECASE))

        if not tool_calls or has_answer:
            # No tool calls - we're done
            all_text += new_text
            turns.append(Turn(
                generated_text=new_text,
                tool_calls=[],
                tool_results=[],
            ))
            break

        # Execute tool calls and build result string
        tool_results = []
        result_text_parts = []

        for tc in tool_calls:
            result = spec.execute_tool(tc)
            tc.result = result
            tool_results.append(str(result))
            result_text_parts.append(f"{tc.tool_name}[{tc.argument}] = {result}")

        # Record this turn
        turns.append(Turn(
            generated_text=new_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
        ))

        # Inject results and continue (format matches system prompt examples)
        result_injection = "\n[Result: " + ", ".join(tool_results) + "]\n"
        all_text += new_text + result_injection

    # Extract final answer from all generated text
    extracted = spec.extract_answer(all_text, [tc for t in turns for tc in t.tool_calls])
    is_correct = extracted == task.correct_answer

    return AgenticResult(
        task=task,
        turns=turns,
        final_text=all_text,
        extracted_answer=extracted,
        is_correct=is_correct,
    )


def print_result(result: AgenticResult, show_trajectory: bool = True):
    """Print a single result."""
    status = "\u2713" if result.is_correct else "\u2717"
    print(f"\n{status} Q: {result.task.question}")
    print(f"   Expected: {result.task.correct_answer}, Got: {result.extracted_answer}")
    print(f"   Turns: {len(result.turns)}, Tool calls: {result.total_tool_calls}")

    if show_trajectory:
        print("   Trajectory:")
        for i, turn in enumerate(result.turns):
            print(f"      --- Turn {i + 1} ---")
            print(f"      {turn.generated_text}")
            for tc, res in zip(turn.tool_calls, turn.tool_results):
                print(f"      [Tool: {tc.tool_name}[{tc.argument}] = {res}]")

"""
Zorplex RL - A synthetic benchmark for training LLMs on multi-step tool use.

Each TaskSpec defines:
- How to generate a task (question + correct answer)
- What tools are available
- How to parse tool calls from model output
- How to evaluate correctness

Provides the core evaluation loop that generates with tool execution.
"""

import operator
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


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
class EvalResult:
    """Result of evaluating a single task."""
    task: Task
    model_output: str
    tool_calls: list[ToolCall]
    extracted_answer: int | str | None
    is_correct: bool


# =============================================================================
# TASK SPECS
# =============================================================================

class TaskSpec(ABC):
    """Base class for task specifications."""

    name: str = "base"
    description: str = "Base task spec"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)
        self._init_data()

    @abstractmethod
    def _init_data(self) -> None:
        """Initialize any data needed for the task (lookup tables, etc)."""
        pass

    @abstractmethod
    def generate_task(self) -> Task:
        """Generate a random task instance."""
        pass

    @abstractmethod
    def get_system_prompt(self, with_hint: bool = False) -> str:
        """Get the system prompt, optionally with tool hints."""
        pass

    @abstractmethod
    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse tool calls from model output."""
        pass

    @abstractmethod
    def execute_tool(self, tool_call: ToolCall) -> str | int:
        """Execute a tool call and return the result."""
        pass

    @abstractmethod
    def extract_answer(self, text: str, tool_calls: list[ToolCall]) -> int | str | None:
        """Extract the final answer from model output."""
        pass

    def evaluate(self, task: Task, model_output: str) -> EvalResult:
        """Evaluate model output against a task."""
        tool_calls = self.parse_tool_calls(model_output)

        # Execute tool calls to get results
        for tc in tool_calls:
            tc.result = self.execute_tool(tc)

        extracted = self.extract_answer(model_output, tool_calls)
        is_correct = extracted == task.correct_answer

        return EvalResult(
            task=task,
            model_output=model_output,
            tool_calls=tool_calls,
            extracted_answer=extracted,
            is_correct=is_correct,
        )


# =============================================================================
# ZORPLEX LOOKUP TABLE (shared across task types)
# =============================================================================

ZORPLEX_WORDS = [
    "apple", "banana", "cat", "dog", "elephant",
    "fish", "grape", "house", "ice", "jungle",
    "kite", "lemon", "moon", "night", "ocean",
    "piano", "queen", "river", "sun", "tree",
    "umbrella", "violet", "water", "xray", "yellow", "zebra",
]


def make_zorplex_table(seed: int = 42) -> dict[str, int]:
    """Create the zorplex lookup table."""
    rng = random.Random(seed)
    return {word: rng.randint(1, 100) for word in ZORPLEX_WORDS}


# =============================================================================
# TASK SPEC: Simple Lookup
# =============================================================================

class SimpleLookupSpec(TaskSpec):
    """Simple single-word zorplex lookup.

    Task: "What is the zorplex value of 'apple'?"
    Tool: LOOKUP[apple] -> 82
    """

    name = "simple"
    description = "Single LOOKUP call"

    def _init_data(self):
        self._table = make_zorplex_table(self.seed)

    def generate_task(self) -> Task:
        word = self._rng.choice(ZORPLEX_WORDS)
        return Task(
            question=f"What is the zorplex value of '{word}'?",
            correct_answer=self._table[word],
            metadata={"word": word},
        )

    def get_system_prompt(self, with_hint: bool = False) -> str:
        base = "You are a helpful assistant."
        if with_hint:
            base += """

You have access to a LOOKUP tool. To find the zorplex value of a word, simply output LOOKUP[word] on its own line. The system will return the value to you. Do NOT write code - just output the tool call directly.

When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the zorplex value of 'cat'?
Assistant: LOOKUP[cat]
[Result: 42]
Assistant: [ANSWER] 42"""
        return base

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        matches = re.findall(r"LOOKUP\[['\"]?(\w+)['\"]?\]", text, re.IGNORECASE)
        return [ToolCall("LOOKUP", m.lower()) for m in matches]

    def execute_tool(self, tool_call: ToolCall) -> int | str:
        if tool_call.tool_name == "LOOKUP":
            value = self._table.get(tool_call.argument.lower())
            return value if value is not None else "UNKNOWN"
        return "UNKNOWN_TOOL"

    def extract_answer(self, text: str, tool_calls: list[ToolCall]) -> int | None:
        # Check for [ANSWER] tag first
        answer_match = re.findall(r'\[ANSWER\]\s*(\d+)', text)
        if answer_match:
            return int(answer_match[-1])

        # If model made correct tool call, use that result
        for tc in tool_calls:
            if tc.result is not None and isinstance(tc.result, int):
                return tc.result

        # Otherwise try to extract a number
        numbers = re.findall(r'\d+', text)
        return int(numbers[-1]) if numbers else None


# =============================================================================
# TASK SPEC: Compositional (multiple lookups + arithmetic)
# =============================================================================

class CompositionalSpec(TaskSpec):
    """Compositional task requiring multiple lookups and arithmetic.

    Task: "What is zorplex('apple') + zorplex('banana')?"
    Tools: LOOKUP[apple] -> 82, LOOKUP[banana] -> 15
    Answer: 82 + 15 = 97
    """

    name = "compositional"
    description = "Multiple LOOKUPs + arithmetic"

    def __init__(self, seed: int = 42, difficulty: str = "easy"):
        self.difficulty = difficulty
        super().__init__(seed)

    def _init_data(self):
        self._table = make_zorplex_table(self.seed)

        if self.difficulty == "easy":
            self._num_words = 2
            self._ops = [("+", operator.add)]
        elif self.difficulty == "medium":
            self._num_words = 2
            self._ops = [("+", operator.add), ("-", operator.sub), ("*", operator.mul)]
        else:
            self._num_words = 3
            self._ops = [("+", operator.add), ("-", operator.sub), ("*", operator.mul)]

    def generate_task(self) -> Task:
        words = self._rng.sample(ZORPLEX_WORDS, self._num_words)
        values = [self._table[w] for w in words]
        op_sym, op_func = self._rng.choice(self._ops)

        if len(words) == 2:
            answer = op_func(values[0], values[1])
            question = f"What is zorplex('{words[0]}') {op_sym} zorplex('{words[1]}')?"
        else:
            answer = op_func(op_func(values[0], values[1]), values[2])
            question = f"What is zorplex('{words[0]}') {op_sym} zorplex('{words[1]}') {op_sym} zorplex('{words[2]}')?"

        return Task(
            question=question,
            correct_answer=answer,
            metadata={"words": words, "values": values, "op": op_sym},
        )

    def get_system_prompt(self, with_hint: bool = False) -> str:
        base = "You are a helpful assistant."
        if with_hint:
            base += """

You have access to a LOOKUP tool. To find the zorplex value of a word, simply output LOOKUP[word] on its own line. The system will return the value to you. Do NOT write code - just output the tool call directly and wait for the result.

For problems requiring multiple lookups, call LOOKUP once, wait for the result, then call it again.

When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is zorplex('cat') + zorplex('dog')?
Assistant: LOOKUP[cat]
[Result: 42]
Assistant: LOOKUP[dog]
[Result: 17]
Assistant: 42 + 17 = 59. [ANSWER] 59"""
        return base

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        matches = re.findall(r"LOOKUP\[['\"]?(\w+)['\"]?\]", text, re.IGNORECASE)
        return [ToolCall("LOOKUP", m.lower()) for m in matches]

    def execute_tool(self, tool_call: ToolCall) -> int | str:
        if tool_call.tool_name == "LOOKUP":
            value = self._table.get(tool_call.argument.lower())
            return value if value is not None else "UNKNOWN"
        return "UNKNOWN_TOOL"

    def extract_answer(self, text: str, tool_calls: list[ToolCall]) -> int | None:
        # Check for [ANSWER] tag first
        answer_match = re.findall(r'\[ANSWER\]\s*(-?\d+)', text)
        if answer_match:
            return int(answer_match[-1])

        # Try to find explicit answer patterns - use findall and take the last match
        patterns = [
            r'(?:the answer is|answer is|answer:)\s*(-?\d+)',
            r'=\s*(-?\d+)\.?\s*(?:The answer|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return int(matches[-1])

        # Fallback: last number in the text
        numbers = re.findall(r'-?\d+', text)
        return int(numbers[-1]) if numbers else None


# =============================================================================
# REGISTRY
# =============================================================================

TASK_SPECS = {
    "simple": SimpleLookupSpec,
    "compositional": CompositionalSpec,
}


def get_spec(name: str, **kwargs) -> TaskSpec:
    """Get a task spec by name."""
    if name not in TASK_SPECS:
        raise ValueError(f"Unknown task spec: {name}. Available: {list(TASK_SPECS.keys())}")
    return TASK_SPECS[name](**kwargs)


# =============================================================================
# AGENTIC EVALUATION
# =============================================================================

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check newly generated tokens
        new_tokens = input_ids[0, self.prompt_length:]
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Stop if we find a complete tool call
        return bool(self.tool_pattern.search(new_text))


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

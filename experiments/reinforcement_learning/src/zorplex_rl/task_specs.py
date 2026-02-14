"""
Task specifications for Zorplex RL experiments.

Each TaskSpec defines:
- How to generate a task (question + correct answer)
- What tools are available
- How to parse tool calls from model output
- How to evaluate correctness
"""

import hashlib
import operator
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
# TASK SPEC: Multi-step (GETKEY -> FETCH chain)
# =============================================================================

class MultiStepSpec(TaskSpec):
    """Multi-step task requiring tool chaining.

    Task: "What is the zorplex value of 'apple'?"
    Tools: GETKEY[apple] -> ZK_ABC123, FETCH[ZK_ABC123] -> 82
    """

    name = "multi_step"
    description = "GETKEY -> FETCH chain"

    def _init_data(self):
        self._table = make_zorplex_table(self.seed)
        self._key_cache: dict[str, str] = {}

    def _make_key(self, word: str) -> str:
        h = hashlib.md5(f"{word}_{self.seed}".encode()).hexdigest()[:6].upper()
        return f"ZK_{h}"

    def generate_task(self) -> Task:
        word = self._rng.choice(ZORPLEX_WORDS)
        key = self._make_key(word)
        value = self._table[word]

        return Task(
            question=f"What is the zorplex value of '{word}'?",
            correct_answer=value,
            metadata={"word": word, "key": key},
        )

    def get_system_prompt(self, with_hint: bool = False) -> str:
        base = "You are a helpful assistant."
        if with_hint:
            base += """

You have access to two tools: GETKEY and FETCH. To find the zorplex value of a word:
1. Output GETKEY[word] to get a key
2. Wait for the result (you'll receive something like ZK_ABC123)
3. Output FETCH[key] using that key to get the actual value

Do NOT write code - just output the tool calls directly and wait for results.

When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the zorplex value of 'apple'?
Assistant: GETKEY[apple]
[Result: ZK_6368CC]
Assistant: FETCH[ZK_6368CC]
[Result: 82]
Assistant: [ANSWER] 82"""
        return base

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        calls = []

        # Parse GETKEY calls
        for match in re.finditer(r"GETKEY\[['\"]?(\w+)['\"]?\]", text, re.IGNORECASE):
            calls.append(ToolCall("GETKEY", match.group(1).lower()))

        # Parse FETCH calls
        for match in re.finditer(r"FETCH\[['\"]?(ZK_\w+)['\"]?\]", text, re.IGNORECASE):
            calls.append(ToolCall("FETCH", match.group(1).upper()))

        return calls

    def execute_tool(self, tool_call: ToolCall) -> int | str:
        if tool_call.tool_name == "GETKEY":
            word = tool_call.argument.lower()
            if word in self._table:
                key = self._make_key(word)
                self._key_cache[key] = word
                return key
            return "UNKNOWN_WORD"

        elif tool_call.tool_name == "FETCH":
            key = tool_call.argument.upper()
            # Try cache first
            word = self._key_cache.get(key)
            if word is None:
                # Try to reverse-lookup
                for w in ZORPLEX_WORDS:
                    if self._make_key(w) == key:
                        word = w
                        break
            if word:
                return self._table[word]
            return "INVALID_KEY"

        return "UNKNOWN_TOOL"

    def extract_answer(self, text: str, tool_calls: list[ToolCall]) -> int | None:
        # Check for [ANSWER] tag first
        answer_match = re.findall(r'\[ANSWER\]\s*(\d+)', text)
        if answer_match:
            return int(answer_match[-1])

        # Check if a FETCH call returned a valid result
        for tc in tool_calls:
            if tc.tool_name == "FETCH" and isinstance(tc.result, int):
                return tc.result

        # Fallback: extract number
        numbers = re.findall(r'\d+', text)
        return int(numbers[-1]) if numbers else None


# =============================================================================
# TASK SPEC: Recursive (follow redirects)
# =============================================================================

class RecursiveSpec(TaskSpec):
    """Recursive task requiring following redirect chains.

    Task: "What is the zorplex value of 'apple'?"
    Tools: LOOKUP[apple] -> "SEE: banana", LOOKUP[banana] -> "SEE: cat", LOOKUP[cat] -> 42

    Max depth is 3 redirects, so at most 4 lookups needed (well under 5 turns).
    """

    name = "recursive"
    description = "Follow LOOKUP redirects"

    def __init__(self, seed: int = 42, max_depth: int = 3):
        self.max_depth = min(max_depth, 3)  # Cap at 3 to stay under 5 turns
        super().__init__(seed)

    def _init_data(self):
        # Base values for some words
        base_values = make_zorplex_table(self.seed)

        # Create redirect chains
        # Split words: some are "terminal" (have values), some are "redirects"
        words = ZORPLEX_WORDS.copy()
        self._rng.shuffle(words)

        # First half are terminals with actual values
        terminals = words[:len(words)//2]
        redirectors = words[len(words)//2:]

        self._table: dict[str, int | str] = {}

        # Terminals get actual values
        for word in terminals:
            self._table[word] = base_values[word]

        # Redirectors point to other words (possibly other redirectors, creating chains)
        # Build chains of varying depths
        for i, word in enumerate(redirectors):
            # Determine chain depth for this word (1 to max_depth)
            depth = (i % self.max_depth) + 1

            # Build a chain: word -> r1 -> r2 -> ... -> terminal
            chain = [word]
            available_redirectors = [w for w in redirectors if w != word and w not in chain]

            for d in range(depth - 1):
                if available_redirectors:
                    next_word = self._rng.choice(available_redirectors)
                    chain.append(next_word)
                    available_redirectors.remove(next_word)

            # Final element points to a terminal
            terminal = self._rng.choice(terminals)
            chain.append(terminal)

            # Create the redirect entries
            for j in range(len(chain) - 1):
                self._table[chain[j]] = f"SEE: {chain[j+1]}"

        # Store terminal values for answer lookup
        self._terminal_values = {w: base_values[w] for w in terminals}

    def _resolve(self, word: str, max_steps: int = 10) -> int | None:
        """Follow redirects to get final value."""
        current = word
        for _ in range(max_steps):
            value = self._table.get(current.lower())
            if value is None:
                return None
            if isinstance(value, int):
                return value
            # It's a redirect
            if value.startswith("SEE: "):
                current = value[5:]
            else:
                return None
        return None  # Too many redirects

    def generate_task(self) -> Task:
        # Pick a word that requires at least one redirect
        redirect_words = [w for w, v in self._table.items() if isinstance(v, str)]
        word = self._rng.choice(redirect_words) if redirect_words else self._rng.choice(ZORPLEX_WORDS)

        correct = self._resolve(word)

        return Task(
            question=f"What is the zorplex value of '{word}'?",
            correct_answer=correct,
            metadata={"word": word, "chain": self._get_chain(word)},
        )

    def _get_chain(self, word: str) -> list[str]:
        """Get the full redirect chain for a word."""
        chain = [word]
        current = word
        for _ in range(10):
            value = self._table.get(current.lower())
            if value is None or isinstance(value, int):
                break
            if value.startswith("SEE: "):
                current = value[5:]
                chain.append(current)
            else:
                break
        return chain

    def get_system_prompt(self, with_hint: bool = False) -> str:
        base = "You are a helpful assistant."
        if with_hint:
            base += """

You have access to a LOOKUP tool. To find the zorplex value of a word, output LOOKUP[word] on its own line.

IMPORTANT: Some lookups return a redirect like "SEE: other_word". When this happens, you must look up the other word. Keep following redirects until you get a number.

Do NOT write code - just output the tool calls directly and wait for results.

When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the zorplex value of 'apple'?
Assistant: LOOKUP[apple]
[Result: SEE: banana]
Assistant: LOOKUP[banana]
[Result: SEE: cat]
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
            if value is None:
                return "UNKNOWN"
            return value
        return "UNKNOWN_TOOL"

    def extract_answer(self, text: str, tool_calls: list[ToolCall]) -> int | None:
        # Check for [ANSWER] tag first
        answer_match = re.findall(r'\[ANSWER\]\s*(\d+)', text)
        if answer_match:
            return int(answer_match[-1])

        # Look for the last LOOKUP that returned an int
        for tc in reversed(tool_calls):
            if tc.result is not None and isinstance(tc.result, int):
                return tc.result

        # Fallback: find "The answer is X" or last number
        patterns = [
            r'(?:the answer is|answer is|answer:)\s*(\d+)',
            r'(?:value of .+ is)\s*(\d+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return int(matches[-1])

        # Last resort: last number
        numbers = re.findall(r'\d+', text)
        return int(numbers[-1]) if numbers else None


# =============================================================================
# REGISTRY
# =============================================================================

TASK_SPECS = {
    "simple": SimpleLookupSpec,
    "compositional": CompositionalSpec,
    "multi_step": MultiStepSpec,
    "recursive": RecursiveSpec,
}


def get_spec(name: str, **kwargs) -> TaskSpec:
    """Get a task spec by name."""
    if name not in TASK_SPECS:
        raise ValueError(f"Unknown task spec: {name}. Available: {list(TASK_SPECS.keys())}")
    return TASK_SPECS[name](**kwargs)


if __name__ == "__main__":
    # Demo each spec
    for name, SpecClass in TASK_SPECS.items():
        print(f"\n{'='*50}")
        print(f"SPEC: {name}")
        print(f"{'='*50}")

        spec = SpecClass(seed=42)
        print(f"Description: {spec.description}")

        for i in range(3):
            task = spec.generate_task()
            print(f"\n  Task {i+1}: {task.question}")
            print(f"  Answer: {task.correct_answer}")
            print(f"  Metadata: {task.metadata}")

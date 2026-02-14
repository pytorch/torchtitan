"""
Zorplex RL - A synthetic benchmark for training LLMs on multi-step tool use.
"""

from .task_specs import (
    TaskSpec,
    Task,
    ToolCall,
    EvalResult,
    SimpleLookupSpec,
    CompositionalSpec,
    MultiStepSpec,
    RecursiveSpec,
    TASK_SPECS,
    get_spec,
    ZORPLEX_WORDS,
    make_zorplex_table,
)

from .evaluate import (
    Turn,
    AgenticResult,
    generate_with_tools,
    print_result,
)

__all__ = [
    "TaskSpec",
    "Task",
    "ToolCall",
    "EvalResult",
    "SimpleLookupSpec",
    "CompositionalSpec",
    "MultiStepSpec",
    "RecursiveSpec",
    "TASK_SPECS",
    "get_spec",
    "ZORPLEX_WORDS",
    "make_zorplex_table",
    "Turn",
    "AgenticResult",
    "generate_with_tools",
    "print_result",
]

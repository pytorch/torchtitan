# Q9: Context / Sequence Parallelism

Find where sequence parallelism (or context parallelism) is handled. Identify the
exact file and line numbers where the sequence dimension of the input tensor is
sharded, gathered, or scattered across ranks during the forward and backward
passes. If the codebase does not support CP/SP, say so and cite the negative
evidence.

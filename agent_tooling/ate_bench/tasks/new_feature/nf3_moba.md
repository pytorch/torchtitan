# NF3: MoBA (Mixture of Block Attention)

Integrate **MoBA** into the base model. MoBA partitions the key/value sequence
into **fixed-size blocks** and routes each query to the **top-`k` blocks** selected
by a learned gate, yielding sub-quadratic attention cost in the sequence length.

Integrating it involves inserting **block-level routing** between the Q/K
projection and the attention computation (`torchtitan/models/common/attention.py`),
composing with the existing attention backend, and **preserving causal masking
inside each selected block** (see `references.md`).

# NF2: DynMoE

Integrate **DynMoE** into the base MoE model. Standard MoE fixes the number of
activated experts `k` per token. DynMoE replaces top-`k` selection with a
**per-expert sigmoid gate and a learned threshold**, so the count of active
experts **varies per token**.

Integrating it involves replacing the framework's top-`k` routing
(`torchtitan/models/common/moe.py`, `TokenChoiceTopKRouter`) with the
gate-and-threshold mechanism, and **reformulating the load-balancing auxiliary
loss for a variable-`k` regime** (see `references.md`).

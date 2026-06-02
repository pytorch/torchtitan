# NF4: MoE++

Integrate **MoE++** into the base MoE model. MoE++ augments a standard MoE expert
pool with **three zero-computation expert types** (`zero`, `copy`, and
`constant`), allowing easy tokens to be routed past the feed-forward layer
entirely.

Integrating it involves **extending the expert pool definition** with the
zero-computation variants, **widening the router output** to cover them
(`torchtitan/models/common/moe.py`), and **adjusting the load-balancing loss** to
prevent the zero experts from absorbing all easy tokens (see `references.md`).

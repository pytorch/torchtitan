# NF1: Differential Attention (Diff Transformer)

Integrate **differential attention** into the base model. Differential attention
replaces a single softmax-attention map with the **difference of two parallel
softmax maps**:

```
Attn(Q, K, V) = ( softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T) ) V
```

where `lambda` is a **learned per-head scalar**. The construction cancels
common-mode attention noise, sharpening which tokens receive mass.

Integrating it involves splitting the per-head Q/K projections in half (into
`Q1,Q2` and `K1,K2`) and introducing `lambda` as a trainable parameter with the
**published initialisation/reparameterisation schedule** (see the reference
implementation in `references.md`).

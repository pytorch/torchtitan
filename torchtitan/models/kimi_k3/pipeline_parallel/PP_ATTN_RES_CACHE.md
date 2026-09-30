# Pipeline parallelism for the block attention residual

Kimi K3 carries a stack of block residuals alongside the hidden state, and a
pipeline hop has to carry that stack too. The split and the schedule are
core's, through `get_module_fqns_per_model_part` and `pipeline_llm`
([`__init__.py`](__init__.py)); the stage that moves the stack is
[`stage.py`](stage.py), the rank cache is [`cache.py`](cache.py), and the
tables that decide what each hop carries are [`layout.py`](layout.py).

## Overview

One virtual stage on one rank, with `attn_res_cache` on and off. Forward in
black, backward in red; the P2P receive and send are the green boxes, solid
forward and hatched backward. The legend in the figure defines every symbol.
With the cache on, the stage sends back
$\nabla\Delta = \nabla B[\Delta] + \text{deposits}$: for each block it
received, the gradient its own backward computes, which includes what came
back in $\nabla\Delta'$, plus the deposits the rank's later stages left for
that block.

![Kimi K3 pipeline stage with the attention residual cache](../../../../assets/images/kimi_k3_pp_attn_res_cache.svg)

## Example

Forward only, with four ranks and two virtual stages each in loop placement,
the shape of the cache-based pipeline figure in the Attention Residuals paper,
and each stage opening one block. A rank holds every block produced at stage
`s - 4` or earlier, because that is where it last ran the micro-batch; a hop
brings the rest. From the second virtual stage on, a hop carries at most
`P - 1 = 3` blocks, where the whole stack would be 4 to 7.

![Forward-only example of the attention residual cache](../../../../assets/images/kimi_k3_pp_attn_res_cache_example.svg)

## The two transports

| `attn_res_cache` | a hop carries | the rank keeps | against a single device |
|---|---|---|---|
| on (default) | hidden `[T, D]` and the blocks the receiving rank has not seen, `[T, Nd, D]` | every block its earlier stages committed or received, per micro-batch, in the rank cache, released after its last stage's forward | the same values; the cached blocks' gradients are summed in another order, so not bitwise |
| off | hidden `[T, D]` and the whole stack `[T, N, D]` | nothing between hops | bitwise |

Plain `1F1B` has one stage per rank, so both transports carry the whole stack
on every hop; with the cache on, the rank cache holds a micro-batch's blocks
only while that stage runs its forward.

With the cache on, the placement must be the loop one, stage `s` on rank
`s mod P`, as in `Interleaved1F1B`; `pipeline_kimi_k3` refuses any other, such
as a V-shaped schedule's.

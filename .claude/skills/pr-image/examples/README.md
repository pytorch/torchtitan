# Examples

Three images drawn for merged torchtitan PRs. They set the text level to aim for (SKILL.md §3 Text, Order): numbers
as labels, each note a sentence, a header and a closing line per panel. Copy a file to your own folder before
rendering it: `render_html.py` writes the PNG next to the HTML.

```text
file              PR and form                               what makes it work      what else it shows
5819_slot_text    #5819 KDA replay state: the slot's        sizes on to-scale bars  numbers as labels, each note a
.html             entries to scale, 64/64/64/128 KiB vs     show the 1.25x cost     sentence, a header and a closing
                  the 256 KiB recurrent state                                       line per panel
5853_layers_text  #5853 full prefill graphs: 28 layers,     the cost is per layer,  per-op CPU in the zoom, the
.html             attention eager in between vs inside      so the strip shows it   stream result vs vLLM
5918_rows_text    #5918 prompt reuse: 16 sample rows,       the saving is a count   token totals per panel, the vLLM
.html             computed vs read from the cache           (16 x 1,024 -> 1,024)   baseline in the footer
```

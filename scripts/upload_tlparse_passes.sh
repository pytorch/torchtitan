#!/bin/bash
# Upload before/after tlparse pass pairs to pastry with diff links.
# Usage: ./upload_tlparse_passes.sh <tlparse_output_dir>

set -euo pipefail

DIR="${1:?Usage: $0 <tlparse_output_dir>}"
ARTIFACT_DIR="$DIR/-_-_-_-"

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "Error: $ARTIFACT_DIR not found"
    exit 1
fi

echo "=== Uploading tlparse pass pairs from $ARTIFACT_DIR ==="
echo ""

# Upload the initial traced graph
traced="$ARTIFACT_DIR/make_fx_graph_traced_0.txt"
if [ -f "$traced" ]; then
    url=$(cat "$traced" | pastry -t "make_fx_graph_traced" -l python -q 2>/dev/null || echo "(pastry failed)")
    echo "make_fx_graph_traced: $url"
fi

# Pair before/after files by pass name.
# Filenames: before_<pass_name>_<N>.txt, after_<pass_name>_<N+1>.txt
# Extract pass name by stripping before_/after_ prefix and trailing _<digits>.txt
for before_file in "$ARTIFACT_DIR"/before_*.txt; do
    [ -f "$before_file" ] || continue

    before_base=$(basename "$before_file" .txt)
    # Strip "before_" prefix and trailing "_<digits>"
    pass_name=$(echo "$before_base" | sed 's/^before_//; s/_[0-9]*$//')

    # Find matching after file
    after_file=$(ls "$ARTIFACT_DIR"/after_${pass_name}_*.txt 2>/dev/null | head -1)
    if [ -z "$after_file" ] || [ ! -f "$after_file" ]; then
        echo "WARN: no after file for $pass_name"
        continue
    fi

    # Skip if no changes
    if diff -q "$before_file" "$after_file" > /dev/null 2>&1; then
        echo "  $pass_name: (no changes)"
        continue
    fi

    before_output=$(cat "$before_file" | pastry -t "before_${pass_name}" -l python -q 2>/dev/null || echo "")
    after_output=$(cat "$after_file" | pastry -t "after_${pass_name}" -l python -q 2>/dev/null || echo "")

    before_paste=$(echo "$before_output" | grep -oP 'P\K[0-9]+' | head -1)
    after_paste=$(echo "$after_output" | grep -oP 'P\K[0-9]+' | head -1)

    if [ -n "$before_paste" ] && [ -n "$after_paste" ]; then
        diff_url="https://www.internalfb.com/intern/diffing/?before_paste_number=${before_paste}&after_paste_number=${after_paste}&regex_remove_pattern=&enable_regex_remove=0&strip_empty_lines=0&line_wrap=0&selected_tab=plain_diff"
        echo "  $pass_name: $diff_url"
    else
        echo "  $pass_name: (pastry upload failed)"
    fi
done

# Upload standalone artifacts
echo ""
echo "=== Standalone artifacts ==="
for f in "$ARTIFACT_DIR"/activation_memory_policy_*.txt \
         "$ARTIFACT_DIR"/fx_codegen_*.txt \
         "$ARTIFACT_DIR"/fx_collectives_analytical_estimation_*.txt \
         "$ARTIFACT_DIR"/fx_compute_nodes_runtime_estimation_*.txt \
         "$ARTIFACT_DIR"/final_graph_after_all_passes_*.txt; do
    [ -f "$f" ] || continue
    basename=$(basename "$f")
    url=$(cat "$f" | pastry -t "$basename" -l python -q 2>/dev/null || echo "(pastry failed)")
    echo "  $basename: $url"
done

#!/bin/bash
# Upload before/after tlparse pass pairs to pastry.
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

# Upload the initial traced graph first
traced="$ARTIFACT_DIR/make_fx_graph_traced_0.txt"
if [ -f "$traced" ]; then
    url=$(cat "$traced" | pastry -t "make_fx_graph_traced_0" -l python -q 2>/dev/null)
    echo "make_fx_graph_traced_0: $url"
fi

# Find all before/after pairs by extracting pass names from before_ files
for before_file in "$ARTIFACT_DIR"/before_*_pass_*.txt; do
    [ -f "$before_file" ] || continue

    basename=$(basename "$before_file")
    # Extract pass name: before_<pass_name>_pass_<N>.txt -> <pass_name>
    pass_name=$(echo "$basename" | sed 's/^before_\(.*\)_pass_[0-9]*\.txt$/\1/')
    before_num=$(echo "$basename" | sed 's/^.*_pass_\([0-9]*\)\.txt$/\1/')

    # Find the matching after file
    after_file=$(ls "$ARTIFACT_DIR"/after_${pass_name}_pass_*.txt 2>/dev/null | head -1)
    if [ -z "$after_file" ]; then
        echo "WARN: no after file for $pass_name, skipping"
        continue
    fi
    after_basename=$(basename "$after_file")
    after_num=$(echo "$after_basename" | sed 's/^.*_pass_\([0-9]*\)\.txt$/\1/')

    if diff -q "$before_file" "$after_file" > /dev/null 2>&1; then
        echo ""
        echo "--- $pass_name --- (no changes, skipping)"
        continue
    fi

    before_output=$(cat "$before_file" | pastry -t "$basename" -l python -q 2>/dev/null)
    after_output=$(cat "$after_file" | pastry -t "$after_basename" -l python -q 2>/dev/null)

    before_paste=$(echo "$before_output" | grep -oP 'P\K[0-9]+' | head -1)
    after_paste=$(echo "$after_output" | grep -oP 'P\K[0-9]+' | head -1)

    diff_url="https://www.internalfb.com/intern/diffing/?before_paste_number=${before_paste}&after_paste_number=${after_paste}&regex_remove_pattern=&enable_regex_remove=0&strip_empty_lines=0&line_wrap=0&selected_tab=plain_diff"

    echo "$pass_name : $diff_url"
done

# Upload standalone artifacts (not before/after pairs)
echo ""
echo "=== Standalone artifacts ==="
for f in "$ARTIFACT_DIR"/activation_memory_policy_*.txt \
         "$ARTIFACT_DIR"/fx_codegen_*.txt \
         "$ARTIFACT_DIR"/fx_collectives_analytical_estimation_*.txt \
         "$ARTIFACT_DIR"/fx_compute_nodes_runtime_estimation_*.txt; do
    [ -f "$f" ] || continue
    basename=$(basename "$f")
    url=$(cat "$f" | pastry -t "$basename" -l python -q 2>/dev/null)
    echo "$basename: $url"
done

#!/bin/bash
# Copy one capture into its agent sandbox, rewrite paths so they point at the repo snapshot (same commit)
# and carry no experiment or tool names, then make the inputs read-only.
#   outputs/finder_substrate/capture/stage.sh <model> <arm>
set -euo pipefail
MODEL=$1
ARM=$2
SRC=/home/bahuang/local/torchtitan/outputs/finder_substrate/captures/${MODEL}_${ARM}
ROOT=/home/bahuang/local/step_analysis
BOX=$ROOT/${MODEL}_${ARM}
REPO=$ROOT/repo
MAIN=/home/bahuang/local/torchtitan
if [ -e $BOX ]; then echo "$BOX exists" >&2; exit 1; fi
mkdir -p $BOX/artifacts $BOX/out
SED=(
  -e "s#$MAIN/outputs/finder_substrate/captures/${MODEL}_${ARM}#$BOX/artifacts#g"
  -e "s#$MAIN/outputs/finder_substrate/capture#$BOX/artifacts/capture_tools#g"
  -e "s#$MAIN/torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts#$BOX/artifacts/capture_tools#g"
  -e "s#$MAIN#$REPO#g"
  -e "s#fusion_finder#capture#g"
  -e "s#finder_substrate#step_analysis#g"
)
cd $SRC
find . -type f ! -path './tb/*' ! -path './structured_logs/*' | while read -r f; do
  mkdir -p "$BOX/artifacts/$(dirname "$f")"
  case "$f" in
    *.gz) zcat "$f" | LC_ALL=C sed "${SED[@]}" | gzip -c > "$BOX/artifacts/$f" ;;
    *) LC_ALL=C sed "${SED[@]}" "$f" > "$BOX/artifacts/$f" ;;
  esac
done
cp /home/bahuang/local/torchtitan/outputs/finder_substrate/hw_profile.json $BOX/hw_profile.json
# Verify: no main-repo path, tool name or experiment name is left.
left=$(
  cd $BOX/artifacts
  { find . -name '*.gz' -exec zgrep -l -E "$MAIN|fusion_finder|finder_substrate" {} + ; \
    grep -rl --exclude='*.gz' -E "$MAIN|fusion_finder|finder_substrate" . ; } 2>/dev/null || true
)
if [ -n "$left" ]; then echo "leftover references in: $left" >&2; exit 1; fi
chmod -R a-w $BOX/artifacts $BOX/hw_profile.json
du -sh $BOX/artifacts

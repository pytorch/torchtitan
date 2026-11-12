#!/bin/bash
# Profiled step-10 runs (CUDA graphs off) of the four models' GraphTrainer recipes, without kernels (base)
# and with the loop's kernels (pack), in each loop's measurement setup. One model per GPU, base then pack.
cd /home/bahuang/local/torchtitan
W=/home/bahuang/local/torchtitan/outputs/agentic_compiler/glue_roofline
A=torchtitan/experiments/graph_trainer/agentic_compiler
S=torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/scripts
fin() { echo "exit $?" >> $1; }
dsv3() {
  $A/dsv3_671b_eager_study/run_graph.sh $1 graph_base_prof $W/dsv3_base
  $A/dsv3_671b_eager_study/run_graph.sh $1 graph_pack_prof $W/dsv3_pack
}
q30() {
  export PACKS=/home/bahuang/local/graph-trainer-kernel-packs
  $A/qwen3_30b_eager_study/run_graph.sh $1 graph_base_prof $W/q30_base
  $A/qwen3_30b_eager_study/run_graph.sh $1 graph_pack_prof $W/q30_pack
}
gptoss() {
  D=$A/gpt_oss_120b_mxfp8_64gpu
  P14=lift_fusion_region_g007,lift_fusion_region_g011,lift_fusion_region_g019,lift_fusion_region_g020,lift_fusion_region_g017,lift_fusion_region_g023,lift_fusion_region_g025,lift_fusion_region_g028,lift_fusion_region_g036,lift_fusion_region_g037,lift_fusion_region_g031,lift_fusion_region_g039,lift_fusion_region_g048,lift_fusion_region_g041
  for r in base:"" pack:$P14; do
    name=gptoss_${r%%:*}; mkdir -p $W/$name
    env NGPU=64 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=$S:$D:. \
      FUSION_PLAN="${r#*:}" RUN=$W/$name python -m torchtitan.train --module loop_recipe --config lifted \
      --comm-backend fake > $W/$name/log.txt 2>&1; fin $W/$name/log.txt
  done
}
q235() {
  D=$A/qwen3_235b_a22b_distmoe_mxfp8
  common="NGPU=256 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=$S:$D:."
  mkdir -p $W/q235_base $W/q235_pack
  env $common FUSION_FINDER_OUT=$W/q235_base python -m torchtitan.train --module loop_recipe --config baseline \
    --comm-backend fake > $W/q235_base/log.txt 2>&1; fin $W/q235_base/log.txt
  env $common FUSION_PLAN=$(cat $D/plans/plan_iter2.txt) LOOP_ITER=$W/q235_pack python -m torchtitan.train \
    --module loop_recipe --config lifted --comm-backend fake > $W/q235_pack/log.txt 2>&1; fin $W/q235_pack/log.txt
}
"$@"

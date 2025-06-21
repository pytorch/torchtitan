python cute_grouped_gemm.py  \
--use_2cta_instrs   \
--ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32 \
--mma_tiler_mn 128,128 --cluster_shape_mn 4,4  \
 --problem_sizes_mnkl "(8192,1280,32,1),(8,4096,1536,1),(640,1280,16,1),(640,512,16,1)"       \
--num_groups 4  --tensormap_update_mode SMEM

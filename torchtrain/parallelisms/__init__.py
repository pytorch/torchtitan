from torchtrain.parallelisms.parallelize_llama import parallelize_llama

models_parallelize_fns = {
    "llama": parallelize_llama,
}

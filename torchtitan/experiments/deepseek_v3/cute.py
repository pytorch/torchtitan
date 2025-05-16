import cutlass
import cutlass.cute as cute
import torch


@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf(">>> Hello world! from Kernel")


@cute.jit
def hello_world():
    cute.printf(">>> Hello world! from CPU")

    cutlass.cuda.initialize_cuda_context()
    kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


print("running with out compile...")
hello_world()
print(f"\n\nrunning with compile...")
compiled = cute.compile(hello_world)
print(f"\n\ncompiled function: {compiled}")
compiled()
print("done")

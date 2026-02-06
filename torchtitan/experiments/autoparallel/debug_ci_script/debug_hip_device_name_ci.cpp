// How to compile and run:
// hipcc hip_device_name.cpp -o hip_device_name
// ./hip_device_name
#include <hip/hip_runtime.h>
#include <stdio.h>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    hipGetErrorString(err));                                    \
            return 1;                                                           \
        }                                                                        \
    } while (0)

// Simulates torch.cuda.get_device_name(None) behavior
void print_current_device_name() {
    int current_device;
    if (hipGetDevice(&current_device) != hipSuccess) {  // This is what torch.cuda.current_device() calls
        fprintf(stderr, "Failed to get current device\n");
        return;
    }

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, current_device) != hipSuccess) {
        fprintf(stderr, "Failed to get device properties\n");
        return;
    }

    printf("Current device %d: %s\n", current_device, prop.name);
}

int main(int argc, char* argv[]) {
    // Demonstrate hipGetDevice / hipSetDevice
    // This is what torch.cuda.get_device_name(None) does internally
    printf("=== Simulating torch.cuda.get_device_name(None) ===\n");
    print_current_device_name();

    return 0;
}

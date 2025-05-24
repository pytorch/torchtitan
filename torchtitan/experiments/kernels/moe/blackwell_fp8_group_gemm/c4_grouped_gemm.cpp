#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

using namespace cute;

// Type definitions for FP8 E4M3 grouped GEMM
using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

// 1SM Configuration
struct MMA1SMConfig {
  using TileShape =
      cute::Shape<cute::_128, cute::_256, cute::Int<128 / sizeof(ElementA)>>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// 2SM Configuration
struct MMA2SMConfig {
  using TileShape =
      cute::Shape<cute::_256, cute::_256, cute::Int<128 / sizeof(ElementA)>>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

template <typename Config> struct GemmBuilder {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, typename Config::TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC *, AlignmentC, ElementC,
          LayoutC *, AlignmentC, typename Config::EpilogueSchedule,
          cutlass::epilogue::fusion::LinearCombination<
              ElementC, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA *, AlignmentA, ElementB,
          LayoutB *, AlignmentB, ElementAccumulator, typename Config::TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Config::KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Gemm1SM = typename GemmBuilder<MMA1SMConfig>::Gemm;
using Gemm2SM = typename GemmBuilder<MMA2SMConfig>::Gemm;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error("CUDA error: " +                                \
                               std::string(cudaGetErrorString(error)));        \
    }                                                                          \
  } while (0)

#define CUTLASS_CHECK(status)                                                  \
  do {                                                                         \
    if (status != cutlass::Status::kSuccess) {                                 \
      throw std::runtime_error("CUTLASS error: " +                             \
                               std::to_string(int(status)));                   \
    }                                                                          \
  } while (0)

template <typename Gemm>
torch::Tensor run_grouped_gemm(const std::vector<torch::Tensor> &a_tensors,
                               const std::vector<torch::Tensor> &b_tensors,
                               const std::vector<torch::Tensor> &c_tensors,
                               const std::vector<float> &alpha_values,
                               const std::vector<float> &beta_values,
                               int max_sm_count) {

  int groups = a_tensors.size();

  // Setup problem shapes
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<torch::Tensor> d_tensors;

  for (int i = 0; i < groups; ++i) {
    int M = a_tensors[i].size(0);
    int K = a_tensors[i].size(1);
    int N = b_tensors[i].size(1);

    problem_sizes_host.push_back({M, N, K});
    d_tensors.push_back(torch::empty_like(c_tensors[i]));
  }

  // Device memory for problem shapes
  cutlass::device_memory::allocation<
      typename ProblemShape::UnderlyingProblemShape>
      problem_sizes(groups);
  problem_sizes.copy_from_host(problem_sizes_host.data());

  // Setup tensor pointers
  std::vector<const ElementA *> ptr_A_host(groups);
  std::vector<const ElementB *> ptr_B_host(groups);
  std::vector<const ElementC *> ptr_C_host(groups);
  std::vector<ElementC *> ptr_D_host(groups);

  for (int i = 0; i < groups; ++i) {
    ptr_A_host[i] = reinterpret_cast<const ElementA *>(a_tensors[i].data_ptr());
    ptr_B_host[i] = reinterpret_cast<const ElementB *>(b_tensors[i].data_ptr());
    ptr_C_host[i] = reinterpret_cast<const ElementC *>(c_tensors[i].data_ptr());
    ptr_D_host[i] = reinterpret_cast<ElementC *>(d_tensors[i].data_ptr());
  }

  cutlass::device_memory::allocation<const ElementA *> ptr_A(groups);
  cutlass::device_memory::allocation<const ElementB *> ptr_B(groups);
  cutlass::device_memory::allocation<const ElementC *> ptr_C(groups);
  cutlass::device_memory::allocation<ElementC *> ptr_D(groups);

  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.copy_from_host(ptr_D_host.data());

  // Setup strides
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  for (int i = 0; i < groups; ++i) {
    auto problem = problem_sizes_host[i];
    int M = cute::get<0>(problem);
    int N = cute::get<1>(problem);
    int K = cute::get<2>(problem);

    stride_A_host.push_back(
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1)));
    stride_B_host.push_back(
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1)));
    stride_C_host.push_back(
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1)));
    stride_D_host.push_back(
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1)));
  }

  cutlass::device_memory::allocation<StrideA> stride_A(groups);
  cutlass::device_memory::allocation<StrideB> stride_B(groups);
  cutlass::device_memory::allocation<StrideC> stride_C(groups);
  cutlass::device_memory::allocation<StrideD> stride_D(groups);

  stride_A.copy_from_host(stride_A_host.data());
  stride_B.copy_from_host(stride_B_host.data());
  stride_C.copy_from_host(stride_C_host.data());
  stride_D.copy_from_host(stride_D_host.data());

  // Setup alpha and beta values
  cutlass::device_memory::allocation<ElementAccumulator> block_alpha(groups);
  cutlass::device_memory::allocation<ElementAccumulator> block_beta(groups);

  std::vector<ElementAccumulator> alpha_host(alpha_values.begin(),
                                             alpha_values.end());
  std::vector<ElementAccumulator> beta_host(beta_values.begin(),
                                            beta_values.end());

  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

  std::vector<ElementAccumulator *> ptr_alpha_host(groups);
  std::vector<ElementAccumulator *> ptr_beta_host(groups);

  for (int i = 0; i < groups; ++i) {
    ptr_alpha_host[i] = block_alpha.get() + i;
    ptr_beta_host[i] = block_beta.get() + i;
  }

  cutlass::device_memory::allocation<ElementAccumulator *> alpha_device(groups);
  cutlass::device_memory::allocation<ElementAccumulator *> beta_device(groups);

  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.copy_from_host(ptr_beta_host.data());

  // Setup kernel hardware info
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      std::min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                   hw_info.device_id),
               max_sm_count);

  // Configure cluster shape based on kernel type
  if constexpr (std::is_same_v<Gemm, Gemm2SM>) {
    hw_info.cluster_shape = dim3(2, 1, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  } else {
    hw_info.cluster_shape = dim3(1, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  }

  // Setup GEMM kernel
  Gemm gemm;

  // Setup fusion arguments for epilogue
  typename Gemm::EpilogueOutputOp::Arguments fusion_args{
      ElementAccumulator(0), // alpha - will use pointer array
      ElementAccumulator(0), // beta - will use pointer array
      alpha_device.get(),    // alpha_ptr_array
      beta_device.get(),     // beta_ptr_array
      cute::make_stride(cute::_0{}, cute::_0{}, 1), // dAlpha - one per group
      cute::make_stride(cute::_0{}, cute::_0{}, 1)  // dBeta - one per group
  };

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler{};
  scheduler.raster_order =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100GroupParams<
          typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions::
          AlongM;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups, problem_sizes.get(), nullptr}, // ProblemShape arguments
      {ptr_A.get(), stride_A.get(), ptr_B.get(),
       stride_B.get()}, // Mainloop arguments
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(),
       stride_D.get()}, // Epilogue arguments
      hw_info,
      scheduler};

  // Get workspace size and allocate
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Initialize and run kernel
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());

  // Wait for completion
  CUDA_CHECK(cudaDeviceSynchronize());

  return torch::stack(d_tensors);
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

torch::Tensor fp8_grouped_gemm_cuda(const std::vector<torch::Tensor> &a_tensors,
                                    const std::vector<torch::Tensor> &b_tensors,
                                    const std::vector<torch::Tensor> &c_tensors,
                                    const std::vector<float> &alpha_values,
                                    const std::vector<float> &beta_values,
                                    bool use_2sm = false,
                                    int max_sm_count = INT_MAX) {

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(
      false,
      "This extension requires CUTLASS_ARCH_MMA_SM100_SUPPORTED to be defined");
#else

  TORCH_CHECK(a_tensors.size() == b_tensors.size(),
              "A and B tensor lists must have same size");
  TORCH_CHECK(a_tensors.size() == c_tensors.size(),
              "A and C tensor lists must have same size");
  TORCH_CHECK(a_tensors.size() == alpha_values.size(),
              "Alpha values must match tensor count");
  TORCH_CHECK(a_tensors.size() == beta_values.size(),
              "Beta values must match tensor count");

  int groups = a_tensors.size();
  TORCH_CHECK(groups > 0, "Must have at least one group");

  // Validate all tensors
  for (int i = 0; i < groups; ++i) {
    TORCH_CHECK(a_tensors[i].is_cuda(), "A tensors must be on CUDA");
    TORCH_CHECK(b_tensors[i].is_cuda(), "B tensors must be on CUDA");
    TORCH_CHECK(c_tensors[i].is_cuda(), "C tensors must be on CUDA");

    // Check dtypes - note: PyTorch might not have direct float8_e4m3fn support
    // yet so we might need to handle this differently
    TORCH_CHECK(a_tensors[i].scalar_type() == torch::kFloat8_e4m3fn ||
                    a_tensors[i].scalar_type() ==
                        torch::kBFloat16, // fallback for testing
                "A tensors must be FP8 E4M3 or BF16");
    TORCH_CHECK(b_tensors[i].scalar_type() == torch::kFloat8_e4m3fn ||
                    b_tensors[i].scalar_type() ==
                        torch::kBFloat16, // fallback for testing
                "B tensors must be FP8 E4M3 or BF16");
    TORCH_CHECK(c_tensors[i].scalar_type() == torch::kFloat16,
                "C tensors must be FP16");

    TORCH_CHECK(a_tensors[i].dim() == 2, "A tensors must be 2D");
    TORCH_CHECK(b_tensors[i].dim() == 2, "B tensors must be 2D");
    TORCH_CHECK(c_tensors[i].dim() == 2, "C tensors must be 2D");
    TORCH_CHECK(a_tensors[i].is_contiguous(), "A tensors must be contiguous");
    TORCH_CHECK(b_tensors[i].is_contiguous(), "B tensors must be contiguous");
    TORCH_CHECK(c_tensors[i].is_contiguous(), "C tensors must be contiguous");

    // Validate dimensions
    int M = a_tensors[i].size(0);
    int K = a_tensors[i].size(1);
    int K_B = b_tensors[i].size(0);
    int N = b_tensors[i].size(1);

    TORCH_CHECK(K == K_B,
                "A and B tensors must have compatible dimensions for group " +
                    std::to_string(i));
    TORCH_CHECK(c_tensors[i].size(0) == M && c_tensors[i].size(1) == N,
                "C tensor shape mismatch for group " + std::to_string(i));
  }

  // Run the appropriate kernel
  if (use_2sm) {
    return run_grouped_gemm<Gemm2SM>(a_tensors, b_tensors, c_tensors,
                                     alpha_values, beta_values, max_sm_count);
  } else {
    return run_grouped_gemm<Gemm1SM>(a_tensors, b_tensors, c_tensors,
                                     alpha_values, beta_values, max_sm_count);
  }

#endif
}

torch::Tensor
fp8_grouped_gemm_cuda_scalar(const std::vector<torch::Tensor> &a_tensors,
                             const std::vector<torch::Tensor> &b_tensors,
                             const std::vector<torch::Tensor> &c_tensors,
                             float alpha = 1.0f, float beta = 0.0f,
                             bool use_2sm = false, int max_sm_count = INT_MAX) {

  int groups = a_tensors.size();
  std::vector<float> alpha_values(groups, alpha);
  std::vector<float> beta_values(groups, beta);

  return fp8_grouped_gemm_cuda(a_tensors, b_tensors, c_tensors, alpha_values,
                               beta_values, use_2sm, max_sm_count);
}

bool is_blackwell_available() {
  cudaDeviceProp props;
  int current_device_id;
  if (cudaGetDevice(&current_device_id) != cudaSuccess) {
    return false;
  }
  if (cudaGetDeviceProperties(&props, current_device_id) != cudaSuccess) {
    return false;
  }
  return (props.major == 10 && props.minor == 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_grouped_gemm", &fp8_grouped_gemm_cuda, "FP8 E4M3 Grouped GEMM",
        py::arg("a_tensors"), py::arg("b_tensors"), py::arg("c_tensors"),
        py::arg("alpha_values"), py::arg("beta_values"),
        py::arg("use_2sm") = false, py::arg("max_sm_count") = INT_MAX);

  m.def("fp8_grouped_gemm_scalar", &fp8_grouped_gemm_cuda_scalar,
        "FP8 E4M3 Grouped GEMM with scalar alpha/beta", py::arg("a_tensors"),
        py::arg("b_tensors"), py::arg("c_tensors"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f, py::arg("use_2sm") = false,
        py::arg("max_sm_count") = INT_MAX);

  m.def("is_blackwell_available", &is_blackwell_available,
        "Check if Blackwell architecture is available");
}

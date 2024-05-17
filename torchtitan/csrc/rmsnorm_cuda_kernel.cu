#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "type_shim.h"
#include "static_switch.h"


template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}


template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}


template<typename U> __device__
void cuRMSOnlineSum(
  const U curr,
  U& sigma2)
{
  sigma2 = sigma2 + curr * curr;
}

template<typename U> __device__
void cuChanRMSOnlineSum(
  const U sigma2B,
  U& sigma2)
{
  sigma2 = sigma2 + sigma2B;
}

template<typename T, typename U>
__device__ void cuWelfordSigma2(
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    U& sigma2,
    U* buf) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensor is contiguous
    // 3) blockDim.y*sizeof(U) shared memory available.
    //
    // compute sum of squares over n2
    sigma2 = U(0);

    if (i1 < n1) {
        // one warp normalizes one n1 index,
        // synchronization is implicit
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        const T* lvals = vals + i1 * n2;

        // unrolled loop
        for (int l = thrx * 4; l < n2; l += 4 * numx) {
            U tmp = U(0);
            for (int k = 0; k < 4 && l + k < n2; ++k) {
                const U curr = static_cast<U>(lvals[l + k]);
                tmp += curr * curr;
            }
            sigma2 += tmp;
        }

        // intra-warp reductions
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sigma2 += __shfl_down_sync(0xffffffff, sigma2, offset);
        }

        // inter-warp reductions
        if (blockDim.y > 1) {
            U* ubuf = (U*)buf;
            if (threadIdx.x == 0) {
                ubuf[threadIdx.y] = sigma2;
            }
            __syncthreads();

            if (threadIdx.y == 0) {
                U tmp = U(0);
                for (int i = 0; i < blockDim.y; ++i) {
                    tmp += ubuf[i];
                }
                sigma2 = tmp;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            buf[0] = sigma2;
        }
        __syncthreads();

        sigma2 = buf[0] / U(n2);
    }
}

template<typename U> U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template<> float rsqrt(float v) {
  return rsqrtf(v);
}
template<> double rsqrt(double v) {
  return rsqrt(v);
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};
} // end namespace


template<typename T, typename U, typename V=T> __global__
void cuApplyRMSNorm(
  V* __restrict__ output_vals,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma)

{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordSigma2(vals,n1,n2,i1,sigma2,buf);

    const T* lvals = vals + i1*n2;
    V* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<V>(c_invvar * curr);
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}


template<typename V> __device__
V clamp_by_magnitude(V curr_gamma, double eps)
{
  const V kMinGamma = V(eps);
  if (curr_gamma >= 0) {
    if (curr_gamma < kMinGamma) {
      return kMinGamma;
    } else {
      return curr_gamma;
    }
  } else {
    if (curr_gamma > -kMinGamma) {
      return -kMinGamma;
    } else {
      return curr_gamma;
    }
  }
}

/*
template<typename T, typename U, typename V, bool MemoryEfficient> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    const double eps,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);

          if (MemoryEfficient) {
            warp_buf2[write_idx] = curr_dout * (c_h) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] = curr_dout * (c_h) * invvar[i1];
          }

      } else {
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      warp_buf2[write_idx] = U(0);
    }
  }
}
*/
template<typename T, typename U, typename V, bool MemoryEfficient>
__device__ void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    const double eps)
{
    int i1 = i1_block + thr_load_row_off;

    if (i1 < i1_end) {
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;

            if (i2 < n2) {
                int load_idx = i1 * n2 + i2;
                int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;

                U c_h = static_cast<U>(input_or_output[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);

                if (MemoryEfficient) {
                    U gamma_val = static_cast<U>(gamma[i2]);
                    U gamma_clamped = static_cast<U>(clamp_by_magnitude(gamma_val, eps));
                    warp_buf2[write_idx] = curr_dout * c_h / gamma_clamped;
                } else {
                    warp_buf2[write_idx] = curr_dout * c_h * invvar[i1];
                }
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            warp_buf2[write_idx] = U(0);
        }
    }
}
/*
template<typename T, typename U, typename V, bool MemoryEfficient> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    const double eps,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);

        if (MemoryEfficient) {
            warp_buf2[write_idx] += curr_dout * (c_h) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] += curr_dout * (c_h) * invvar[i1];
          }

      }
    }
  }
}

template<typename T, typename U, typename V, bool MemoryEfficient>
__device__ void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const double eps)
{
    int i1 = i1_block + thr_load_row_off;

    if (i1 < i1_end) {
        extern __shared__ U shared_mem[];
        U* c_h = shared_mem;
        U* curr_dout = shared_mem + blockDim.y;

        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            if (i2 < n2) {
                int load_idx = i1 * n2 + i2;
                c_h[k] = static_cast<U>(input_or_output[load_idx]);
                curr_dout[k] = static_cast<U>(dout[load_idx]);
            }
        }

        __syncthreads();

        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;

            if (i2 < n2) {
                if (MemoryEfficient) {
                    U gamma_val = static_cast<U>(gamma[i2]);
                    U gamma_clamped = static_cast<U>(clamp_by_magnitude(gamma_val, eps));
                    warp_buf2[write_idx] += curr_dout[k] * c_h[k] / gamma_clamped;
                } else {
                    warp_buf2[write_idx] += curr_dout[k] * c_h[k] * invvar[i1];
                }
            }
        }
    }
}
*/
template<typename T, typename U, typename V, bool MemoryEfficient>
__device__ __forceinline__ void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf2,
    const T* __restrict__ input_or_output,
    const V* __restrict__ dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const double eps)
{
    //constexpr double eps = 1e-8;  // Example value

    int i1 = i1_block + thr_load_row_off;

    if (i1 < i1_end) {
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;

            if (i2 < n2) {
                int load_idx = i1 * n2 + i2;
                U c_h = static_cast<U>(__ldg(&input_or_output[load_idx]));
                U curr_dout = static_cast<U>(__ldg(&dout[load_idx]));

                if (MemoryEfficient) {
                    U gamma_val = static_cast<U>(__ldg(&gamma[i2]));
                    U gamma_clamped = static_cast<U>(clamp_by_magnitude(gamma_val, eps));
                    warp_buf2[write_idx] += curr_dout * c_h / gamma_clamped;
                } else {
                    warp_buf2[write_idx] += curr_dout * c_h * __ldg(&invvar[i1]);
                }
            }
        }
    }
}

template<typename T, typename U, typename V, bool MemoryEfficient> __global__
void cuComputePartGradGammaBeta(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    U* part_grad_gamma,
    U* part_grad_beta,
    const double eps,
    bool rms_only)
{
    const int numsegs_n1 = (n1+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs<T, U, V, MemoryEfficient>(i1_beg,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input_or_output,dout,i1_end,n2,mean,invvar,gamma,beta,eps);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y;  i1_block < i1_end;  i1_block+=blockDim.y*blockDim.y) {
      cuLoadAddStridedInputs<T, U, V, MemoryEfficient>(i1_block,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf2,input_or_output,dout,i1_end,n2,invvar,gamma,eps);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0;  k < blockDim.y;  ++k) {
      int row1 = threadIdx.y + k*blockDim.y;
      int idx1 = row1*row_stride + threadIdx.x;
      if (!rms_only) {
        acc1 += warp_buf1[idx1];
      }
      acc2 += warp_buf2[idx1];
    }

    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2;  offset > 1;  offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + offset;
        int idx1 = row1*row_stride + threadIdx.x;
        int idx2 = row2*row_stride + threadIdx.x;

        warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1*row_stride + threadIdx.x;
      int idx2 = row2*row_stride + threadIdx.x;

      part_grad_gamma[blockIdx.y*n2+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename U, typename V> __global__
void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const U* part_grad_beta,
    const int part_size,
    const int n1,
    const int n2,
    V* grad_gamma,
    V* grad_beta,
    bool rms_only)
{
    // sum partial gradients for gamma and beta
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
      // each warp does sequential reductions until reduced part_size is num_warps
      int num_warp_reductions = part_size / blockDim.y;
      U sum_gamma = U(0);
      U sum_beta = U(0);
      const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
      const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
      for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
        sum_gamma += part_grad_gamma_ptr[warp_offset*n2];

      }
      // inter-warp reductions
      const int nbsize3 = blockDim.x * blockDim.y / 2;
      for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[write_idx] = sum_gamma;

        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
          const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_gamma += buf[read_idx];

        }
        __syncthreads();
      }
      // write out fully summed gradients
      if (threadIdx.y == 0) {
        grad_gamma[i2] = sum_gamma;
      }
    }
}


template<typename T, typename U, typename V, bool MemoryEfficient> __global__
void cuComputeGradInput(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* gamma,
    const V* beta,
    T* grad_input,
    const double eps,
    bool rms_only)
{
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const T* k_h = input_or_output + i1*n2;
    const V* k_dout = dout + i1*n2;
    const U c_invvar = invvar[i1];
    const U c_mean = !MemoryEfficient ? mean[i1] : 0.;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_h[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss * gamma[l+k];
            if (MemoryEfficient) {
              sum_loss2 += c_loss * (c_h - beta[l+k]);
            } else {
              sum_loss2 += c_loss * gamma[l+k] * (c_h - c_mean) * c_invvar;
            }
          } else {
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * gamma[l+k] * (c_h) * c_invvar;
            }
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss * gamma[l];
          if (MemoryEfficient) {
            sum_loss2 += c_loss * (c_h - beta[l]);
          } else {
            sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
          }
        } else {
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * gamma[l] * (c_h) * c_invvar;
          }
        }
      }
    } else {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_h[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss;
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
            }
          } else {
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * (c_h) * c_invvar;
            }
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss;
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
          }
        } else {
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * (c_h) * c_invvar;
          }
        }
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer();
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss2 = buf[2*threadIdx.x+1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1*n2;
    if (gamma != NULL) {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        const U k_gamma = static_cast<U>(clamp_by_magnitude(gamma[l], eps));
        U f_grad_input = fH * c_loss * k_gamma;

          if (MemoryEfficient) {
            f_grad_input -= c_h / k_gamma * sum_loss2;
          } else {
            f_grad_input -= c_h * c_invvar * sum_loss2;
          }

        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          if (MemoryEfficient) {
            f_grad_input -= c_h * sum_loss2;
          } else {
            f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
          }
        } else {
          if (MemoryEfficient) {
            f_grad_input -= c_h * sum_loss2;
          } else {
            f_grad_input -= c_h * c_invvar * sum_loss2;
          }
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}


template<typename T, typename U, typename V=T>
void HostApplyRMSNorm(
    V* output,
    U* invvar,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const V* gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32,4,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
    int nshared =
        threads.y > 1 ?
            threads.y*sizeof(U)+(threads.y/2)*sizeof(U) :
            0;
    cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, invvar, input, n1, n2, U(epsilon), gamma);
}


void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "rms_norm_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        HostApplyRMSNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          output->DATA_PTR<scalar_t_out>(),
          invvar->DATA_PTR<accscalar_t>(),
          input->DATA_PTR<scalar_t_in>(),
          n1,n2,
          epsilon,
          gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL);
      )
}


template<typename T, typename U=float, typename V=T>
void HostRMSNormGradient(
    const V* dout,
    const U* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    const V* gamma,
    double epsilon,
    T* grad_input,
    V* grad_gamma,
    bool memory_efficient)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL) {
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(U);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      // note (mkozuki): I can hard code part_grad_gamma's dtype as float given that
      // the `cuda_layer_norm_gradient` doesn't support double.
      const auto part_grad_dtype =
        (input_or_output->scalar_type() == at::ScalarType::Half || input_or_output->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input_or_output->scalar_type();
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input_or_output->options().dtype(part_grad_dtype));
      BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{
        auto kernel = &cuComputePartGradGammaBeta<T, U, V, MemoryEfficient>;
        kernel<<<blocks2, threads2, nshared2, stream>>>(
                        dout,
                        input_or_output->DATA_PTR<T>(),
                        n1,n2,
                        invvar, /* unused */
                        invvar,
                        U(epsilon),
                        gamma,
                        gamma, /* unused */
                        part_grad_gamma.DATA_PTR<U>(),
                        part_grad_gamma.DATA_PTR<U>(), /* unused */
                        epsilon,
                        true);
      });


      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(U);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_gamma.DATA_PTR<U>(), /* unused */
                      part_size,
                      n1,n2,
                      grad_gamma,
                      grad_gamma, /* unused */
                      true);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
    const dim3 threads1(32,4,1);
    int nshared =
            threads1.y > 1 ?
            threads1.y*threads1.x*sizeof(U) :
            0;
    BOOL_SWITCH(memory_efficient, MemoryEfficient, [&] {
      auto kernel = cuComputeGradInput<T, U, V, MemoryEfficient>;
      kernel<<<blocks1, threads1, nshared, stream>>>(
              dout,
              input_or_output->DATA_PTR<T>(),
              n1,n2,
              invvar, /* unused */
              invvar,
              U(epsilon),
              gamma,
              gamma, /* unused */
              grad_input,
              epsilon,
              true);
    });
}


void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    bool memory_efficient)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      input_or_output->scalar_type(),
      gamma == NULL ? input_or_output->scalar_type() :  gamma->scalar_type(),
      "cuComputeGradInputRMS",
      using accscalar_t = at::acc_type<scalar_t_in, true>;
      HostRMSNormGradient(
        dout->DATA_PTR<scalar_t_out>(),
        invvar->DATA_PTR<accscalar_t>(),
        input_or_output,
        n1,n2,
            // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
            // if gamma Tensor is NULL on input.
        gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
        epsilon,
        grad_input->DATA_PTR<scalar_t_in>(),
        gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
        memory_efficient);
    )
}
/*
void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    bool memory_efficient) {
    using namespace at;

    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input_or_output->scalar_type(),
        gamma == nullptr ? input_or_output->scalar_type() : gamma->scalar_type(),
        "cuComputeGradInputRMS",
        [&] {
            using accscalar_t = acc_type<scalar_t_in, true>;
            HostRMSNormGradient(
                dout->DATA_PTR<scalar_t_out>(),
                invvar->DATA_PTR<accscalar_t>(),
                input_or_output,
                n1,
                n2,
                gamma != nullptr ? gamma->DATA_PTR<scalar_t_out>() : nullptr,
                epsilon,
                grad_input->DATA_PTR<scalar_t_in>(),
                grad_gamma != nullptr ? grad_gamma->DATA_PTR<scalar_t_out>() : nullptr,
                memory_efficient);
        }
    );
}
*/
/*
template<typename AccType>
__device__ void cuRMSOnlineSum(const AccType curr, AccType& sigma2) {
  sigma2 = sigma2 + curr * curr;
}


template<typename AccType>
__device__ void cuChanRMSOnlineSum(const AccType sigma2B, AccType& sigma2) {
  sigma2 = sigma2 + sigma2B;
}

// updated to remove Mu since we only care about RMSNorm
template<typename InputType, typename AccType>
__device__ void cuWelfordSigma2(
    const InputType* __restrict__ vals, const int n1, const int n2, const int i1,
    AccType& sigma2, AccType* buf) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensor is contiguous
    // 3) blockDim.y*sizeof(AccType) shared memory available.
    //
    // compute sum of squares over n2
    sigma2 = AccType(0);
    if (i1 < n1) {
        // one warp normalizes one n1 index,
        // synchronization is implicit
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        const InputType* lvals = vals + i1*n2;
        int l = 4*thrx;
        for (; l+3 < n2; l+=4*numx) {
            for (int k = 0; k < 4; ++k) {
                AccType curr = static_cast<AccType>(__ldg(&lvals[l+k]));
                cuRMSOnlineSum<AccType>(curr, sigma2);
            }
        }
        for (; l < n2; ++l) {
            AccType curr = static_cast<AccType>(__ldg(&lvals[l]));
            cuRMSOnlineSum<AccType>(curr, sigma2);
        }
        // intra-warp reductions
        #pragma unroll
        for (int l = 0; l <= 4; ++l) {
            int srcLaneB = (threadIdx.x+(1<<l))&31;
            AccType sigma2B = WARP_SHFL(sigma2, srcLaneB);
            cuChanRMSOnlineSum<AccType>(sigma2B, sigma2);
        }
        // threadIdx.x == 0 has correct values for each warp
        // inter-warp reductions
        if (blockDim.y > 1) {
            AccType* ubuf = static_cast<AccType*>(buf);
            for (int offset = blockDim.y/2; offset > 0; offset /= 2) {
                // upper half of warps write to shared
                if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
                    const int wrt_y = threadIdx.y - offset;
                    ubuf[wrt_y] = sigma2;
                }
                __syncthreads();
                // lower half merges
                if (threadIdx.x == 0 && threadIdx.y < offset) {
                    AccType sigma2B = ubuf[threadIdx.y];
                    cuChanRMSOnlineSum<AccType>(sigma2B, sigma2);
                }
                __syncthreads();
            }
            // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct value
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                ubuf[0] = sigma2;
            }
            __syncthreads();
            sigma2 = ubuf[0] / AccType(n2);
        } else {
            sigma2 = WARP_SHFL(sigma2 / AccType(n2), 0);
        }
    }
}

// rsqrts

template<typename AccType>
__device__ __forceinline__ AccType rsqrt(AccType v) {
  return AccType(1) / sqrt(v);
}

template<>
__device__ __forceinline__ float rsqrt<float>(float v) {
  return rsqrtf(v);
}

template<>
__device__ __forceinline__ double rsqrt<double>(double v) {
  return rsqrt(v);
}

// shared memory struct
namespace {
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
    __device__ float* getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory<double> {
    __device__ double* getPointer() {
        extern __shared__ double s_double[];
        return s_double;
    }
};
}

template<typename InputType, typename AccType, typename OutputType>
__device__ void cuApplyRMSNorm_(
  OutputType* __restrict__ output_vals,
  AccType* __restrict__ invvar,
  const InputType* __restrict__ vals,
  const int n1,
  const int n2,
  const AccType epsilon,
  const OutputType* __restrict__ gamma)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<AccType> shared;
    AccType* buf = shared.getPointer();
    AccType sigma2;
    cuWelfordSigma2(vals, n1, n2, i1, sigma2, buf);

    const InputType* lvals = vals + i1*n2;
    OutputType* ovals = output_vals + i1*n2;
    AccType c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      for (int i = thrx;  i < n2;  i += numx) {
        AccType curr = static_cast<AccType>(__ldg(&lvals[i]));
        ovals[i] = gamma[i] * static_cast<OutputType>(c_invvar * curr);
      }
    } else {
      for (int i = thrx;  i < n2;  i += numx) {
        AccType curr = static_cast<AccType>(__ldg(&lvals[i]));
        ovals[i] = static_cast<OutputType>(c_invvar * curr);
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template<typename InputType, typename AccType, typename OutputType=InputType>
__global__ void cuApplyRMSNorm(
  OutputType* __restrict__ output_vals,
  AccType* __restrict__ invvar,
  const InputType* __restrict__ vals,
  const int n1,
  const int n2,
  const AccType epsilon,
  const OutputType* __restrict__ gamma)
{
  cuApplyRMSNorm_<InputType, AccType, OutputType>(output_vals, invvar, vals, n1, n2, epsilon, gamma);
}

template<typename OutputType>
__device__ OutputType clamp_by_magnitude(OutputType curr_gamma, double eps)
{
  const OutputType kMinGamma = OutputType(eps);
  return abs(curr_gamma) < kMinGamma ? (curr_gamma < 0 ? -kMinGamma : kMinGamma) : curr_gamma;
}


template<typename InputType, typename AccType, typename OutputType, bool MemoryEfficient>
__device__ void cuLoadAddStridedInputsRMSOnly(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, AccType* warp_buf2,
    const InputType* input_or_output, const OutputType* dout, const int i1_end, const int n2,
    const AccType* __restrict__ invvar, const OutputType* __restrict__ gamma, const double eps) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                AccType c_h = static_cast<AccType>(__ldg(&input_or_output[load_idx]));
                AccType curr_dout = static_cast<AccType>(__ldg(&dout[load_idx]));
                if (MemoryEfficient) {
                    warp_buf2[write_idx] += curr_dout * c_h / static_cast<AccType>(clamp_by_magnitude(gamma[i2], eps));
                } else {
                    warp_buf2[write_idx] += curr_dout * c_h * invvar[i1];
                }
            }
        }
    }
}

template<typename InputType, typename AccType, typename OutputType, bool MemoryEfficient>
__device__ void cuLoadWriteStridedInputsRMSOnly(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, AccType* warp_buf2,
    const InputType* input_or_output, const OutputType* dout, const int i1_end, const int n2,
    const AccType* __restrict__ invvar, const OutputType* __restrict__ gamma, const double eps) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                AccType c_h = static_cast<AccType>(__ldg(&input_or_output[load_idx]));
                AccType curr_dout = static_cast<AccType>(__ldg(&dout[load_idx]));
                if (MemoryEfficient) {
                    warp_buf2[write_idx] = curr_dout * c_h / static_cast<AccType>(clamp_by_magnitude(gamma[i2], eps));
                } else {
                    warp_buf2[write_idx] = curr_dout * c_h * invvar[i1];
                }
            } else {
                warp_buf2[write_idx] = AccType(0);
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            warp_buf2[write_idx] = AccType(0);
        }
    }
}
template<typename InputType, typename AccType, typename OutputType, bool MemoryEfficient>
__global__ void cuComputeGradInput(
    const OutputType* __restrict__ dout,
    const InputType* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const AccType* __restrict__ invvar,
    AccType epsilon,
    const OutputType* gamma,
    InputType* grad_input,
    const double eps)
{
    for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
        AccType sum_loss2 = AccType(0);
        const InputType* k_h = input_or_output + i1*n2;
        const OutputType* k_dout = dout + i1*n2;
        const AccType c_invvar = invvar[i1];
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        if (gamma != NULL) {
            int l = 4*thrx;
            for (;  l+3 < n2;  l+=4*numx) {
                #pragma unroll
                for (int k = 0;  k < 4;  ++k) {
                    const AccType c_h = static_cast<AccType>(__ldg(&k_h[l+k]));
                    const AccType c_loss = static_cast<AccType>(__ldg(&k_dout[l+k]));
                    if (MemoryEfficient) {
                        sum_loss2 += c_loss * c_h;
                    } else {
                        sum_loss2 += c_loss * gamma[l+k] * c_h * c_invvar;
                    }
                }
            }
            for (;  l < n2;  ++l) {
                const AccType c_h = static_cast<AccType>(__ldg(&k_h[l]));
                const AccType c_loss = static_cast<AccType>(__ldg(&k_dout[l]));
                if (MemoryEfficient) {
                    sum_loss2 += c_loss * c_h;
                } else {
                    sum_loss2 += c_loss * gamma[l] * c_h * c_invvar;
                }
            }
        } else {
            int l = 4*thrx;
            for (;  l+3 < n2;  l+=4*numx) {
                #pragma unroll
                for (int k = 0;  k < 4;  ++k) {
                    const AccType c_h = static_cast<AccType>(__ldg(&k_h[l+k]));
                    const AccType c_loss = static_cast<AccType>(__ldg(&k_dout[l+k]));
                    if (MemoryEfficient) {
                        sum_loss2 += c_loss * c_h;
                    } else {
                        sum_loss2 += c_loss * c_h * c_invvar;
                    }
                }
            }
            for (;  l < n2;  ++l) {
                const AccType c_h = static_cast<AccType>(__ldg(&k_h[l]));
                const AccType c_loss = static_cast<AccType>(__ldg(&k_dout[l]));
                if (MemoryEfficient) {
                    sum_loss2 += c_loss * c_h;
                } else {
                    sum_loss2 += c_loss * c_h * c_invvar;
                }
            }
        }
        // intra-warp reductions
        #pragma unroll
        for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
            sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
        }
        // inter-warp reductions
        if (blockDim.y > 1) {
            SharedMemory<AccType> shared;
            AccType* buf = shared.getPointer();
            for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
                // upper half of warps write to shared
                if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
                    const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                    buf[wrt_i] = sum_loss2;
                }
                __syncthreads();
                // lower half merges
                if (threadIdx.y < offset) {
                    const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
                    sum_loss2 += buf[read_i];
                }
                __syncthreads();
            }
            if (threadIdx.y == 0) {
                buf[threadIdx.x] = sum_loss2;
            }
            __syncthreads();
            if (threadIdx.y != 0) {
                sum_loss2 = buf[threadIdx.x];
            }
        }
        // all threads now have the sum over l
        const AccType fH = static_cast<AccType>(n2);
        const AccType term1 = (AccType(1) / fH) * c_invvar;
        InputType* k_grad_input = grad_input + i1*n2;
        if (gamma != NULL) {
            for (int l = thrx;  l < n2;  l+=numx) {
                const AccType c_h = static_cast<AccType>(__ldg(&k_h[l]));
                const AccType c_loss = static_cast<AccType>(__ldg(&k_dout[l]));
                const AccType k_gamma = static_cast<AccType>(clamp_by_magnitude(gamma[l], eps));
                AccType f_grad_input = fH * c_loss * k_gamma;
                if (MemoryEfficient) {
                    f_grad_input -= c_h / k_gamma * sum_loss2;
                } else {
                    f_grad_input -= c_h * c_invvar * sum_loss2;
                }
                f_grad_input *= term1;
                k_grad_input[l] = static_cast<InputType>(f_grad_input);
            }
        } else {
            for (int l = thrx;  l < n2;  l+=numx) {
                const AccType c_h = static_cast<AccType>(__ldg(&k_h[l]));
                const AccType c_loss = static_cast<AccType>(__ldg(&k_dout[l]));
                AccType f_grad_input = fH * c_loss;
                if (MemoryEfficient) {
                    f_grad_input -= c_h * sum_loss2;
                } else {
                    f_grad_input -= c_h * c_invvar * sum_loss2;
                }
                f_grad_input *= term1;
                k_grad_input[l] = static_cast<InputType>(f_grad_input);
            }
        }
        // prevent race where buf is written again before reads are done
        __syncthreads();
    }
}
template<typename GradType, typename SumType>
__global__ void cuComputeGradGamma(
    const GradType* __restrict__ part_grad_gamma,
    const int part_size,
    const int n1,
    const int n2,
    SumType* __restrict__ grad_gamma)
{
    // sum partial gradients for gamma
    SharedMemory<GradType> shared;
    GradType* buf = shared.getPointer();
    const int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        const int num_warp_reductions = part_size / blockDim.y;
        GradType sum_gamma = GradType(0);
        const GradType* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
        for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
            sum_gamma += static_cast<GradType>(__ldg(&part_grad_gamma_ptr[warp_offset*n2]));
        }
        // inter-warp reductions
        for (int offset = blockDim.y/2; offset >= 1; offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
            }
            __syncthreads();
        }
        // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
        }
    }
}
template<typename InputType, typename AccType, typename OutputType, bool MemoryEfficient>
__global__ void cuComputePartGradGammaBeta(
    const OutputType* __restrict__ dout,
    const InputType* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const AccType* __restrict__ invvar,
    const OutputType* __restrict__ gamma,
    const double eps,
    AccType* __restrict__ part_grad_gamma)
{
    const int blockSize = blockDim.y * blockDim.y;
    const int numsegs_n1 = (n1 + blockSize - 1) / blockSize;
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockSize;
    const int i1_beg_plus_one = (blockIdx.y + 1) * segs_per_block * blockSize;
    const int i1_end = std::min(i1_beg_plus_one, n1);
    const int row_stride = blockDim.x + 1;
    const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
    const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;

    SharedMemory<AccType> shared;
    AccType* buf = shared.getPointer();
    AccType* warp_buf2 = buf;

    cuLoadWriteStridedInputsRMSOnly<InputType, AccType, OutputType, MemoryEfficient>(
        i1_beg, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
        warp_buf2, input_or_output, dout, i1_end, n2, invvar, gamma, eps);

    for (int i1_block = i1_beg + blockSize; i1_block < i1_end; i1_block += blockSize) {
        cuLoadAddStridedInputsRMSOnly<InputType, AccType, OutputType, MemoryEfficient>(
            i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
            warp_buf2, input_or_output, dout, i1_end, n2, invvar, gamma, eps);
    }
    __syncthreads();

    // inter-warp reductions
    // sum within each warp
    AccType acc2 = AccType(0);
    for (int k = 0; k < blockDim.y; ++k) {
        int row1 = threadIdx.y + k * blockDim.y;
        int idx1 = row1 * row_stride + threadIdx.x;
        acc2 += warp_buf2[idx1];
    }
    warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
    __syncthreads();

    // sum all warps
    #pragma unroll
    for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int row1 = threadIdx.y;
            int row2 = threadIdx.y + offset;
            int idx1 = row1 * row_stride + threadIdx.x;
            int idx2 = row2 * row_stride + threadIdx.x;
            warp_buf2[idx1] += warp_buf2[idx2];
        }
        __syncthreads();
    }

    const int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + 1;
        int idx1 = row1 * row_stride + threadIdx.x;
        int idx2 = row2 * row_stride + threadIdx.x;
        part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename InputType, typename AccType, typename OutputType>
void HostRMSNormGradient(
    const OutputType* dout,
    const AccType* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    const OutputType* gamma,
    double epsilon,
    InputType* grad_input,
    OutputType* grad_gamma,
    bool memory_efficient)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != nullptr) {
        constexpr int part_size = 16;
        constexpr int threads_x = 32;
        constexpr int threads_y = 4;
        const dim3 threads2(threads_x, threads_y, 1);
        const dim3 blocks2((n2 + threads_x - 1) / threads_x, part_size, 1);
        constexpr int nshared2_a = 2 * sizeof(AccType) * threads_y * threads_y * (threads_x + 1);
        constexpr int nshared2_b = threads_x * threads_y * sizeof(AccType);
        constexpr int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;

        const auto part_grad_dtype = (input_or_output->scalar_type() == at::ScalarType::Half ||
                                      input_or_output->scalar_type() == at::ScalarType::BFloat16)
                                     ? at::ScalarType::Float
                                     : input_or_output->scalar_type();

        at::Tensor part_grad_gamma = at::empty({part_size, n2}, input_or_output->options().dtype(part_grad_dtype));

        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&] {
            auto kernel = &cuComputePartGradGammaBeta<InputType, AccType, OutputType, MemoryEfficient>;
            kernel<<<blocks2, threads2, nshared2, stream>>>(
                dout,
                input_or_output->DATA_PTR<InputType>(),
                n1, n2,
                invvar,
                gamma,
                epsilon,
                part_grad_gamma.DATA_PTR<AccType>());
        });

        constexpr int threads3_x = 32;
        constexpr int threads3_y = 8;
        const dim3 threads3(threads3_x, threads3_y, 1);
        const dim3 blocks3((n2 + threads3_x - 1) / threads3_x, 1, 1);
        constexpr int nshared3 = threads3_x * threads3_y * sizeof(AccType);

        cuComputeGradGamma<<<blocks3, threads3, nshared3, stream>>>(
            part_grad_gamma.DATA_PTR<AccType>(),
            part_size,
            n1, n2,
            grad_gamma);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
    constexpr int threads1_x = 32;
    constexpr int threads1_y = 4;
    const dim3 threads1(threads1_x, threads1_y, 1);
    constexpr int nshared = threads1_y > 1 ? threads1_y * threads1_x * sizeof(AccType) : 0;

    BOOL_SWITCH(memory_efficient, MemoryEfficient, [&] {
        auto kernel = &cuComputeGradInput<InputType, AccType, OutputType, MemoryEfficient>;
        kernel<<<blocks1, threads1, nshared, stream>>>(
            dout,
            input_or_output->DATA_PTR<InputType>(),
            n1, n2,
            invvar,
            epsilon,
            gamma,
            grad_input,
            epsilon);
    });
}
void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    bool memory_efficient)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input_or_output->scalar_type(), gamma == nullptr ? input_or_output->scalar_type() : gamma->scalar_type(),
        "cuComputeGradInputRMS", [&] {
            using accscalar_t = at::acc_type<scalar_t_in, true>;
            HostRMSNormGradient(
                dout->DATA_PTR<scalar_t_out>(),
                invvar->DATA_PTR<accscalar_t>(),
                input_or_output,
                n1, n2,
                gamma != nullptr ? gamma->DATA_PTR<scalar_t_out>() : nullptr,
                epsilon,
                grad_input->DATA_PTR<scalar_t_in>(),
                gamma != nullptr ? grad_gamma->DATA_PTR<scalar_t_out>() : nullptr,
                memory_efficient);
        }
    );
}

template<typename InputType, typename AccType, typename OutputType>
void HostApplyRMSNorm(
    OutputType* output,
    AccType* invvar,
    const InputType* input,
    int n1,
    int n2,
    double epsilon,
    const OutputType* gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads_x = 32;
    constexpr int threads_y = 4;
    const dim3 threads(threads_x, threads_y, 1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
    constexpr int nshared = threads_y > 1 ? threads_y * sizeof(AccType) + (threads_y / 2) * sizeof(AccType) : 0;
    cuApplyRMSNorm<InputType, AccType, OutputType><<<blocks, threads, nshared, stream>>>(
        output, invvar, input, n1, n2, AccType(epsilon), gamma);
}

void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "rms_norm_cuda_kernel", [&] {
            using accscalar_t = acc_type<scalar_t_in, true>;
            HostApplyRMSNorm<scalar_t_in, accscalar_t, scalar_t_out>(
                output->DATA_PTR<scalar_t_out>(),
                invvar->DATA_PTR<accscalar_t>(),
                input->DATA_PTR<scalar_t_in>(),
                n1, n2,
                epsilon,
                gamma != nullptr ? gamma->DATA_PTR<scalar_t_out>() : nullptr);
        }
    );
}
*/

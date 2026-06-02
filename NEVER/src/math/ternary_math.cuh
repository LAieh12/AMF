#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

namespace never {
namespace math {

// Quantize FP32 to Ternary {-1, 0, 1}
__device__ __forceinline__ int8_t quantize_ternary(float val, float threshold = 0.5f) {
    if (val > threshold) return 1;
    if (val < -threshold) return -1;
    return 0;
}

// Ternary dot product: optimized kernel for {-1, 0, 1}
// Avoids costly multiplications
template <int BLOCK_SIZE>
__global__ void ternary_dot_product(const float* __restrict__ inputs, 
                                    const int8_t* __restrict__ ternary_weights, 
                                    float* __restrict__ outputs, 
                                    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    
    float local_sum = 0.0f;
    
    if (idx < N) {
        int8_t weight = ternary_weights[idx];
        if (weight == 1) {
            local_sum = inputs[idx];
        } else if (weight == -1) {
            local_sum = -inputs[idx];
        }
    }
    
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Standard reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write out the result for this block
    if (threadIdx.x == 0) {
        atomicAdd(outputs, shared_sum[0]);
    }
}

// Applies a ternary masked linear transformation
// f(x) = W_ternary * x + b
__global__ void ternary_linear_layer(const float* __restrict__ input,
                                     const int8_t* __restrict__ weight,
                                     const float* __restrict__ bias,
                                     float* __restrict__ output,
                                     int in_features,
                                     int out_features) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_features) return;

    float sum = (bias != nullptr) ? bias[out_idx] : 0.0f;

    for (int i = 0; i < in_features; ++i) {
        int8_t w = weight[out_idx * in_features + i];
        if (w == 1) sum += input[i];
        else if (w == -1) sum -= input[i];
    }

    output[out_idx] = sum;
}

} // namespace math
} // namespace never

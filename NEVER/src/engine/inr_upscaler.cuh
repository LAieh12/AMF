#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace never {
namespace engine {

// Native upscaling by evaluating the INR at the target resolution space
// Instead of interpolating pixels, we query the continuous function f(x,y)
__global__ void inr_upscale_evaluate_kernel(float* __restrict__ d_weights, // Simplified MLP weights
                                            float* __restrict__ d_bias,
                                            float* __restrict__ output_image, // RGB
                                            int target_width,
                                            int target_height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= target_width || py >= target_height) return;

    // Normalize coordinates to [-1, 1] continuous space
    float nx = (2.0f * px / target_width) - 1.0f;
    float ny = (2.0f * py / target_height) - 1.0f;

    // Simplified 1-layer evaluation for demonstration
    // In a full implementation, this uses the deep MLP evaluated at (nx, ny)
    float r = nx * d_weights[0] + ny * d_weights[1] + d_bias[0];
    float g = nx * d_weights[2] + ny * d_weights[3] + d_bias[1];
    float b = nx * d_weights[4] + ny * d_weights[5] + d_bias[2];

    // Sigmoid or clamp activation
    r = max(0.0f, min(1.0f, r));
    g = max(0.0f, min(1.0f, g));
    b = max(0.0f, min(1.0f, b));

    int out_idx = (py * target_width + px) * 3;
    output_image[out_idx] = r;
    output_image[out_idx + 1] = g;
    output_image[out_idx + 2] = b;
}

} // namespace engine
} // namespace never

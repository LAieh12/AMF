#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include "core/inr_base.cuh"

namespace never {
namespace engine {

// Manages the final output combination of Static RAM backgrounds and Dynamic VRAM events
class OutputHandler {
private:
    int m_width;
    int m_height;
    
    // Final composited RGB buffer (ready for FFmpeg pipe or OpenGL interop)
    uint8_t* d_final_frame_rgb;

public:
    OutputHandler(int width, int height) : m_width(width), m_height(height) {
        cudaMalloc(&d_final_frame_rgb, m_width * m_height * 3 * sizeof(uint8_t));
        cudaMemset(d_final_frame_rgb, 0, m_width * m_height * 3 * sizeof(uint8_t));
    }

    ~OutputHandler() {
        if(d_final_frame_rgb) cudaFree(d_final_frame_rgb);
    }

    uint8_t* get_final_buffer() { return d_final_frame_rgb; }
    int get_width() const { return m_width; }
    int get_height() const { return m_height; }
};

// Combiner Kernel: Merges the Static INR Background with the Dynamic AMF frame
// Driven by the DeltaOcclusion XOR mask
__global__ void composite_output_kernel(const core::OutputFeatures* __restrict__ static_cache,
                                        const uint8_t* __restrict__ delta_mask,
                                        const float* __restrict__ dynamic_rgb, // Output from AMF vector decoder
                                        uint8_t* __restrict__ final_rgb,
                                        int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int idx = py * width + px;
    int rgb_idx = idx * 3;

    // Check Occlusion XOR Mask (Has this pixel been moved by AMF dynamics?)
    if (delta_mask[idx] == 1) {
        // Dynamic Branch Takes Priority (Entity rendered over background)
        // Note: dynamic_rgb is assumed to be [0.0, 1.0] from the INR Upscaler
        final_rgb[rgb_idx]     = (uint8_t)(dynamic_rgb[rgb_idx] * 255.0f);
        final_rgb[rgb_idx + 1] = (uint8_t)(dynamic_rgb[rgb_idx + 1] * 255.0f);
        final_rgb[rgb_idx + 2] = (uint8_t)(dynamic_rgb[rgb_idx + 2] * 255.0f);
    } else {
        // Static Branch (RAM Cache)
        final_rgb[rgb_idx]     = (uint8_t)(static_cache[idx].r * 255.0f);
        final_rgb[rgb_idx + 1] = (uint8_t)(static_cache[idx].g * 255.0f);
        final_rgb[rgb_idx + 2] = (uint8_t)(static_cache[idx].b * 255.0f);
    }
}

} // namespace engine
} // namespace never

#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "engine/amf_world_model.cuh"

namespace never {
namespace engine {

// Vectorial decoder: turns latent S(t+1) into a coherent RGBA frame.
// It is intentionally not photorealistic; it preserves pose, identity and
// physical consistency so the presenter can composite or encode video.
__global__ void amf_decode_latent_to_rgba_kernel(
    NeverLatentState latent,
    uint8_t* __restrict__ rgba,
    int width,
    int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float x = (2.0f * (float)px / fmaxf(1.0f, (float)(width - 1))) - 1.0f;
    float y = (2.0f * (float)py / fmaxf(1.0f, (float)(height - 1))) - 1.0f;
    float dx = x - latent.x;
    float dy = y - latent.y;
    float r = fmaxf(latent.radius, 0.025f);
    float blob = expf(-(dx * dx + dy * dy) / (2.0f * r * r));
    float vx_color = fminf(fmaxf(0.5f + 0.16f * latent.vx, 0.0f), 1.0f);
    float vy_color = fminf(fmaxf(0.5f + 0.16f * latent.vy, 0.0f), 1.0f);
    float style = fminf(fmaxf(0.25f + 0.5f * latent.shape_code, 0.0f), 1.0f);

    int idx = (py * width + px) * 4;
    rgba[idx + 0] = (uint8_t)(255.0f * fminf(fmaxf(blob, 0.0f), 1.0f));
    rgba[idx + 1] = (uint8_t)(255.0f * fminf(fmaxf(blob * vx_color, 0.0f), 1.0f));
    rgba[idx + 2] = (uint8_t)(255.0f * fminf(fmaxf(blob * vy_color, 0.0f), 1.0f));
    rgba[idx + 3] = (uint8_t)(255.0f * fminf(fmaxf(blob * style, 0.0f), 1.0f));
}

} // namespace engine
} // namespace never

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace never {
namespace engine {

// Ternary State Space Model for propagating Visual Inertia
struct TernarySSMState {
    float* d_state_vector; // h(t)
    int state_dim;

    void allocate(int dim) {
        state_dim = dim;
        cudaMalloc(&d_state_vector, state_dim * sizeof(float));
        cudaMemset(d_state_vector, 0, state_dim * sizeof(float));
    }
    
    void free() {
        if(d_state_vector) cudaFree(d_state_vector);
    }
    
    void reset() {
        cudaMemset(d_state_vector, 0, state_dim * sizeof(float));
    }
};

// Delta Occlusion Calculator
// Optimizes PCIe bandwidth by transmitting only depth changes via XOR
class DeltaOcclusion {
private:
    int m_width;
    int m_height;
    
    // Depth buffers (acting as the occlusion mask)
    float* d_occlusion_current; // Occlusion(t)
    float* d_occlusion_prev;    // Occlusion(t-1)
    
    uint8_t* d_delta_mask;      // DeltaOcclusion(t) = Occlusion(t) XOR Occlusion(t-1)

public:
    DeltaOcclusion(int width, int height) : m_width(width), m_height(height) {
        size_t size = m_width * m_height;
        cudaMalloc(&d_occlusion_current, size * sizeof(float));
        cudaMalloc(&d_occlusion_prev, size * sizeof(float));
        cudaMalloc(&d_delta_mask, size * sizeof(uint8_t));
        
        cudaMemset(d_occlusion_current, 0, size * sizeof(float));
        cudaMemset(d_occlusion_prev, 0, size * sizeof(float));
        cudaMemset(d_delta_mask, 0, size * sizeof(uint8_t));
    }

    ~DeltaOcclusion() {
        if(d_occlusion_current) cudaFree(d_occlusion_current);
        if(d_occlusion_prev) cudaFree(d_occlusion_prev);
        if(d_delta_mask) cudaFree(d_delta_mask);
    }
    
    float* get_current_occlusion() { return d_occlusion_current; }
    uint8_t* get_delta_mask() { return d_delta_mask; }
    
    // Call this at the end of the frame
    void cycle_buffers() {
        // Swap pointers is faster, but copying works for testing
        // pointer swap logic is usually implemented up higher in the rendering loop
        // Standard approach: cudaMemcpy(d_occlusion_prev, d_occlusion_current, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
};

// Kernel: Compute Delta Occlusion Mask
// Converts depth changes into a sparse bitmask for transmission
__global__ void compute_delta_occlusion_kernel(const float* __restrict__ occ_current,
                                               const float* __restrict__ occ_prev,
                                               uint8_t* __restrict__ delta_mask,
                                               int num_pixels,
                                               float threshold = 0.001f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float diff = abs(occ_current[idx] - occ_prev[idx]);
    
    // Logical XOR proxy based on depth change threshold
    if (diff > threshold) {
        delta_mask[idx] = 1;
    } else {
        delta_mask[idx] = 0;
    }
}

} // namespace engine
} // namespace never

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace never {
namespace engine {

// Implementation of Sparse Boundary Loss
// L_total = \sum_{\Delta Occlusion} || \nabla f_{static} - \nabla f_{dynamic} ||^2 + \lambda * L_{audio}
class SparseBoundaryLoss {
private:
    float m_lambda_audio;
    float* d_loss_output; // Single float for the reduced loss

public:
    SparseBoundaryLoss(float lambda_audio = 0.1f) : m_lambda_audio(lambda_audio) {
        cudaMalloc(&d_loss_output, sizeof(float));
        cudaMemset(d_loss_output, 0, sizeof(float));
    }

    ~SparseBoundaryLoss() {
        if(d_loss_output) cudaFree(d_loss_output);
    }
    
    float* get_loss_ptr() { return d_loss_output; }
    float get_lambda_audio() const { return m_lambda_audio; }
};

// Computes the sparse loss only where Delta Occlusion is active (1)
// Optimizes training by focusing on the ~5% of the frame where interaction occurs
__global__ void compute_sparse_boundary_loss_kernel(const uint8_t* __restrict__ delta_mask,
                                                    const float* __restrict__ grad_f_static_x,
                                                    const float* __restrict__ grad_f_static_y,
                                                    const float* __restrict__ grad_f_dyn_x,
                                                    const float* __restrict__ grad_f_dyn_y,
                                                    float* __restrict__ total_loss,
                                                    int num_pixels,
                                                    float lambda_audio,
                                                    float audio_loss) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    // Only compute where interaction occurred (delta_mask == 1)
    if (delta_mask[idx] == 1) {
        float dx = grad_f_static_x[idx] - grad_f_dyn_x[idx];
        float dy = grad_f_static_y[idx] - grad_f_dyn_y[idx];
        
        float squared_norm = (dx * dx) + (dy * dy);
        
        // Atomic add to accumulate the global loss
        atomicAdd(total_loss, squared_norm);
    }
    
    // Thread 0 adds the audio penalty component
    if (idx == 0) {
        atomicAdd(total_loss, lambda_audio * audio_loss);
    }
}

} // namespace engine
} // namespace never

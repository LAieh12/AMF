#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace never {
namespace engine {

// Ambient Branch (Audio) - Context: VRAM/RAM
// Parallel independent SSM conditioned by the Temporal Anchor and Kinetic Energy
class AmbientAudioSSM {
private:
    int m_state_dim;
    float* d_audio_state;     // h(t) for audio
    float* d_audio_buffer;    // Output audio buffer (PCM samples)
    int m_buffer_size;

public:
    AmbientAudioSSM(int state_dim, int buffer_size = 48000) 
        : m_state_dim(state_dim), m_buffer_size(buffer_size) {
        cudaMalloc(&d_audio_state, m_state_dim * sizeof(float));
        cudaMalloc(&d_audio_buffer, m_buffer_size * sizeof(float));
        
        clear_history();
    }
    
    ~AmbientAudioSSM() {
        if(d_audio_state) cudaFree(d_audio_state);
        if(d_audio_buffer) cudaFree(d_audio_buffer);
    }

    // Explicit clear_history() required on each scene transition to avoid state leaks
    void clear_history() {
        cudaMemset(d_audio_state, 0, m_state_dim * sizeof(float));
        cudaMemset(d_audio_buffer, 0, m_buffer_size * sizeof(float));
    }
    
    // Returns pointer to the generated PCM data
    float* get_audio_buffer() { return d_audio_buffer; }
};

// Generates Diegetic Audio from the temporal derivative: df_dynamic / dt
// Conditioned by the kinetic energy (Ek) from the Temporal Anchor
__global__ void compute_diegetic_audio_kernel(const float* __restrict__ f_dyn_current,
                                              const float* __restrict__ f_dyn_prev,
                                              const float* __restrict__ kinetic_energy_Ek,
                                              float* __restrict__ audio_out,
                                              int num_samples,
                                              float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    // Temporal derivative: (f(t) - f(t-1)) / dt
    float derivative = (f_dyn_current[idx] - f_dyn_prev[idx]) / dt;
    
    // Scale by kinetic energy from the temporal anchor
    float ek = kinetic_energy_Ek[0]; // Assuming global Ek scalar for the scene/entity
    
    // Output audio sample
    audio_out[idx] = derivative * ek;
}

// 12ms Crossfade for clean scene transitions (prevent digital clicks)
__global__ void audio_crossfade_12ms_kernel(const float* __restrict__ audio_scene_A,
                                            const float* __restrict__ audio_scene_B,
                                            float* __restrict__ audio_out,
                                            int crossfade_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= crossfade_samples) return;

    // Linear crossfade over the 12ms window
    float t = (float)idx / (float)crossfade_samples; 
    
    audio_out[idx] = audio_scene_A[idx] * (1.0f - t) + audio_scene_B[idx] * t;
}

} // namespace engine
} // namespace never

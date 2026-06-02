#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace never {
namespace core {

// Dual Anchor Space structure
// Aligns visuals (x,y) with temporal audio (t, E_k)
struct DualAnchorState {
    // Visual anchor (frozen for uniform latent space)
    float visual_x;
    float visual_y;
    
    // Temporal anchor
    float time_t;       // current time parameter
    float kinetic_Ek;   // audio/kinetic energy

    __host__ __device__ DualAnchorState() 
        : visual_x(0.0f), visual_y(0.0f), time_t(0.0f), kinetic_Ek(0.0f) {}
};

// Global device state pointer
extern __device__ DualAnchorState* d_global_anchor;

// Host-side controller for the Dual Anchor
class DualAnchorController {
private:
    DualAnchorState* d_state;
    DualAnchorState h_state;
    
public:
    DualAnchorController() {
        cudaMalloc(&d_state, sizeof(DualAnchorState));
        update_device();
    }
    
    ~DualAnchorController() {
        if (d_state) cudaFree(d_state);
    }
    
    void update_visual(float x, float y) {
        h_state.visual_x = x;
        h_state.visual_y = y;
    }
    
    void update_temporal(float t, float Ek) {
        h_state.time_t = t;
        h_state.kinetic_Ek = Ek;
    }
    
    void sync() {
        update_device();
    }
    
    DualAnchorState* get_device_ptr() const { return d_state; }

private:
    void update_device() {
        cudaMemcpy(d_state, &h_state, sizeof(DualAnchorState), cudaMemcpyHostToDevice);
    }
};

} // namespace core
} // namespace never

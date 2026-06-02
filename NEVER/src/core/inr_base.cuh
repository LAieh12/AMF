#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <vector>

namespace never {
namespace core {

// Coordinate point in continuous space (Visual Anchor)
struct Vector2f {
    float x;
    float y;
};

struct OutputFeatures {
    float r, g, b;
    float depth; // Occlusion mask proxy
};

// Static INR Cache resides in RAM/VRAM boundary.
// It stores evaluated points for static background with a dirty_mask
class StaticINRCache {
private:
    int m_width;
    int m_height;
    
    OutputFeatures* d_cache;
    uint8_t* d_dirty_mask; // 1 byte per tile/pixel to indicate invalidation

    OutputFeatures* h_cache; // CPU side
    uint8_t* h_dirty_mask;   // CPU side

public:
    StaticINRCache(int width, int height) : m_width(width), m_height(height) {
        size_t grid_size = m_width * m_height;
        
        // Allocate Device memory
        cudaMalloc(&d_cache, grid_size * sizeof(OutputFeatures));
        cudaMalloc(&d_dirty_mask, grid_size * sizeof(uint8_t));
        
        // Allocate Host memory (pinned for fast transfers if needed, but standard malloc for now)
        cudaMallocHost(&h_cache, grid_size * sizeof(OutputFeatures));
        cudaMallocHost(&h_dirty_mask, grid_size * sizeof(uint8_t));

        // Initialize completely dirty
        memset(h_dirty_mask, 1, grid_size * sizeof(uint8_t));
        cudaMemset(d_dirty_mask, 1, grid_size * sizeof(uint8_t));
    }

    ~StaticINRCache() {
        if(d_cache) cudaFree(d_cache);
        if(d_dirty_mask) cudaFree(d_dirty_mask);
        if(h_cache) cudaFreeHost(h_cache);
        if(h_dirty_mask) cudaFreeHost(h_dirty_mask);
    }

    OutputFeatures* get_device_cache() { return d_cache; }
    uint8_t* get_device_dirty_mask() { return d_dirty_mask; }
    
    int get_width() const { return m_width; }
    int get_height() const { return m_height; }

    // Host side manipulation
    void invalidate_region(int x, int y, int width, int height) {
        for(int cy = y; cy < y + height && cy < m_height; ++cy) {
            for(int cx = x; cx < x + width && cx < m_width; ++cx) {
                h_dirty_mask[cy * m_width + cx] = 1;
            }
        }
    }
    
    void sync_masks_to_device() {
        cudaMemcpy(d_dirty_mask, h_dirty_mask, m_width * m_height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
};

// Simple dense block for the MLP/INR
template<int IN_DIM, int OUT_DIM>
struct DenseLayer {
    float* d_weights; // shape [OUT_DIM, IN_DIM]
    float* d_bias;    // shape [OUT_DIM]
    
    void allocate() {
        cudaMalloc(&d_weights, OUT_DIM * IN_DIM * sizeof(float));
        cudaMalloc(&d_bias, OUT_DIM * sizeof(float));
    }
    
    void free() {
        cudaFree(d_weights);
        cudaFree(d_bias);
    }
};

} // namespace core
} // namespace never

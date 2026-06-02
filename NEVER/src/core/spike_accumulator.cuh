#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

namespace never {
namespace core {

// Spike Event structure
struct SpikeEvent {
    uint32_t x : 12;
    uint32_t y : 12;
    int32_t polarity : 2; // -1, 0, 1
    uint32_t channel : 6;
};

// Lock-free Ring Buffer for Spike events (MAX_PENDING = 4096)
template <uint32_t MAX_PENDING = 4096>
class SpikeAccumulator {
private:
    SpikeEvent* d_buffer;
    uint32_t* d_head;
    uint32_t* d_tail;

public:
    SpikeAccumulator() {
        cudaMalloc(&d_buffer, MAX_PENDING * sizeof(SpikeEvent));
        cudaMalloc(&d_head, sizeof(uint32_t));
        cudaMalloc(&d_tail, sizeof(uint32_t));
        cudaMemset(d_head, 0, sizeof(uint32_t));
        cudaMemset(d_tail, 0, sizeof(uint32_t));
    }

    ~SpikeAccumulator() {
        if (d_buffer) cudaFree(d_buffer);
        if (d_head) cudaFree(d_head);
        if (d_tail) cudaFree(d_tail);
    }

    // Kernel-side insertion (lock-free using atomicAdd)
    __device__ void push_spike(uint32_t x, uint32_t y, int8_t polarity, uint8_t channel) {
        uint32_t index = atomicAdd(d_tail, 1) % MAX_PENDING;
        
        SpikeEvent event;
        event.x = x;
        event.y = y;
        event.polarity = polarity;
        event.channel = channel;
        
        d_buffer[index] = event;
    }

    // Kernel-side extraction (lock-free)
    // Returns true if event retrieved, false if buffer empty
    __device__ bool pop_spike(SpikeEvent& out_event) {
        uint32_t current_head = *d_head;
        uint32_t current_tail = *d_tail;

        if (current_head == current_tail) {
            return false; // Vector is empty
        }

        uint32_t index = atomicAdd(d_head, 1) % MAX_PENDING;
        out_event = d_buffer[index];
        return true;
    }

    // Utility: Reset the buffer
    void reset() {
        cudaMemset(d_head, 0, sizeof(uint32_t));
        cudaMemset(d_tail, 0, sizeof(uint32_t));
    }
};

} // namespace core
} // namespace never

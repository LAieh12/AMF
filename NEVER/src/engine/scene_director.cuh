#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/inr_base.cuh"

namespace never {
namespace engine {

// Director class managing scene transitions and Double Buffering
class SceneDirector {
private:
    core::StaticINRCache* m_active_scene;
    core::StaticINRCache* m_prefetch_scene;
    
    bool m_is_transitioning;
    int m_transition_frame_countdown;

public:
    SceneDirector(core::StaticINRCache* initial_scene, core::StaticINRCache* secondary_buffer) {
        m_active_scene = initial_scene;
        m_prefetch_scene = secondary_buffer;
        m_is_transitioning = false;
        m_transition_frame_countdown = 0;
    }
    
    // 1. Pre-fetching: Director loads Scene B (simulated here)
    void begin_transition(int frames_ahead = 24) {
        m_is_transitioning = true;
        m_transition_frame_countdown = frames_ahead;
        // In reality, this would trigger background loading to m_prefetch_scene
    }
    
    // Call per frame
    bool update() {
        if (m_is_transitioning) {
            m_transition_frame_countdown--;
            
            if (m_transition_frame_countdown <= 0) {
                execute_swap();
                return true; // Transition just completed this frame
            }
        }
        return false;
    }
    
    core::StaticINRCache* get_active_scene() { return m_active_scene; }

private:
    // 2. Atomic: Pointer-swap of static cache
    void execute_swap() {
        core::StaticINRCache* temp = m_active_scene;
        m_active_scene = m_prefetch_scene;
        m_prefetch_scene = temp;
        
        m_is_transitioning = false;
        // Note: $\Delta$Occlusion reset and Audio Crossfade are managed in main loop
    }
};

} // namespace engine
} // namespace never

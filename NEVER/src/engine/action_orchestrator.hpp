#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include "engine/amf_world_model.cuh"

namespace never {
namespace engine {

class ExternalActionOrchestrator {
public:
    // Local deterministic fallback. The Python orchestrator owns the real API
    // call path; this keeps the CUDA mock executable runnable without secrets.
    NeverActionVector prompt_to_action(const std::string& prompt, int frame) const {
        std::string lower = prompt;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return (char)std::tolower(c);
        });

        NeverActionVector action{};
        action.force_x = 0.10f * std::sin(0.11f * frame);
        action.force_y = 0.05f;
        action.camera_yaw_deg = 0.0f;
        action.style_hold = 1.0f;

        if (lower.find("jump") != std::string::npos || lower.find("salt") != std::string::npos) {
            action.force_y += 0.85f;
        }
        if (lower.find("left") != std::string::npos || lower.find("izquierda") != std::string::npos) {
            action.force_x -= 0.55f;
        }
        if (lower.find("right") != std::string::npos || lower.find("derecha") != std::string::npos) {
            action.force_x += 0.55f;
        }
        if (lower.find("camera") != std::string::npos || lower.find("camara") != std::string::npos) {
            action.camera_yaw_deg = 45.0f;
        }
        return action;
    }
};

} // namespace engine
} // namespace never

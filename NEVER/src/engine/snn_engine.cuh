#pragma once

// Phase 10c retirement notice:
// The old SNN dynamic branch has been removed from the active N.E.V.E.R. path.
// Dynamic state now flows through:
//
//   prompt -> external action vector -> visual latent S(t)
//          -> AMFWorldModelRuntime -> S(t+1) -> AMF vector decoder
//
// Keep this file only as a migration marker so old include paths fail softly
// during review rather than silently instantiating the obsolete LIF engine.

#include "engine/amf_world_model.cuh"

namespace never {
namespace engine {

struct RetiredSNNEngine {
    static const char* replacement() {
        return "AMFWorldModelRuntime in engine/amf_world_model.cuh";
    }
};

} // namespace engine
} // namespace never

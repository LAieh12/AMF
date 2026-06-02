#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "engine/action_orchestrator.hpp"
#include "engine/amf_vector_decoder.cuh"
#include "engine/amf_world_model.cuh"

using namespace never::engine;

int main(int argc, char** argv) {
    std::cout << "=======================================" << std::endl;
    std::cout << " N.E.V.E.R. PHASE 10C INFERENCE CORE" << std::endl;
    std::cout << " SNN branch retired. AMF latent world model active." << std::endl;
    std::cout << "=======================================\n" << std::endl;

    std::string prompt = "anime character jumps, camera rotates 45 degrees";
    if (argc > 1) {
        prompt = argv[1];
    }

    AMFWorldModelRuntime amf;
    amf.load_warm_export("../data/phase10a_warm_amf.json");
    amf.freeze_identity_from_geometry(2048, 1.0f);

    ExternalActionOrchestrator orchestrator;
    NeverLatentState state{0.0f, 0.0f, 0.0f, 0.0f, 0.065f, 1.0f, 1.0f, 1.0f};

    constexpr int width = 256;
    constexpr int height = 256;
    uint8_t* d_presenter_frame = nullptr;
    cudaMalloc(&d_presenter_frame, width * height * 4 * sizeof(uint8_t));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    for (int frame = 0; frame < 60; ++frame) {
        NeverActionVector action = orchestrator.prompt_to_action(prompt, frame);

        // Phase 10c core:
        // prompt -> action A(t)
        // frame(t) -> encoder -> S(t)     (Python path owns real encoder)
        // AMF(S(t), A(t)) -> S(t+1)
        // S(t+1) -> vector decoder -> frame
        state = amf.predict_host(state, action);
        amf_decode_latent_to_rgba_kernel<<<grid, block>>>(state, d_presenter_frame, width, height);

        if (frame % 15 == 0) {
            std::cout << "[Frame " << frame << "] S=(" << state.x << ", " << state.y
                      << ", " << state.vx << ", " << state.vy << ")"
                      << " A=(" << action.force_x << ", " << action.force_y << ")"
                      << " camera_yaw=" << action.camera_yaw_deg << std::endl;
        }
    }

    cudaDeviceSynchronize();
    cudaFree(d_presenter_frame);

    std::cout << "\nAMF inference loop complete. Presenter buffer generated." << std::endl;
    return 0;
}

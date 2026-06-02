#include <iostream>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

#include "engine/action_orchestrator.hpp"
#include "engine/amf_vector_decoder.cuh"
#include "engine/amf_world_model.cuh"

using namespace never::engine;

NeverLatentState make_initial_state() {
    NeverLatentState state{};
    state.x = 0.0f;
    state.y = 0.0f;
    state.vx = 0.0f;
    state.vy = 0.0f;
    state.radius = 0.065f;
    state.shape_code = 1.0f;
    state.wall_x = never_wall_feature(state.x);
    state.wall_y = never_wall_feature(state.y);
    return state;
}

NeverLatentState simulate_real_never_latent(
    const NeverLatentState& state,
    const NeverActionVector& action,
    int frame
) {
    constexpr float dt = 0.08f;
    float ax = fminf(fmaxf(action.force_x, -1.0f), 1.0f);
    float ay = fminf(fmaxf(action.force_y, -1.0f), 1.0f);
    float camera_drive = fminf(fmaxf(action.camera_yaw_deg / 90.0f, -1.0f), 1.0f);
    float style_drive = fminf(fmaxf(action.style_hold, 0.0f), 1.0f);

    float next_vx = state.vx + dt * (1.25f * ax);
    float next_vy = state.vy + dt * (1.25f * ay - 0.85f);
    float delta[4];
    delta[0] = dt * next_vx + 0.020f * sinf(0.31f * frame) + 0.010f * camera_drive;
    delta[1] = dt * next_vy + 0.018f * fmaxf(0.0f, sinf(0.23f * frame));
    delta[2] = (next_vx - state.vx) + 0.060f * camera_drive + 0.025f * cosf(0.17f * frame);
    delta[3] = (next_vy - state.vy) + 0.085f * cosf(0.23f * frame) * style_drive;
    NeverLatentState out = never_apply_delta(state, delta);
    out.vx *= 0.992f;
    out.vy *= 0.985f;
    out.wall_x = never_wall_feature(out.x);
    out.wall_y = never_wall_feature(out.y);
    return out;
}

NeverLatentState make_probe_next(const NeverLatentState& state, const NeverActionVector& action) {
    NeverLatentState out = simulate_real_never_latent(state, action, 777);
    out.x = fminf(fmaxf(out.x + 0.16f, -1.05f), 1.05f);
    out.y = fminf(fmaxf(out.y - 0.13f, -1.05f), 1.05f);
    out.vx = fminf(fmaxf(out.vx - 0.46f, -3.0f), 3.0f);
    out.vy = fminf(fmaxf(out.vy + 0.52f, -3.0f), 3.0f);
    out.wall_x = never_wall_feature(out.x);
    out.wall_y = never_wall_feature(out.y);
    return out;
}

int main(int argc, char** argv) {
    std::cout << "=======================================" << std::endl;
    std::cout << " N.E.V.E.R. AMF WORLD MODEL CORE" << std::endl;
    std::cout << " Phase 10c: prompt -> action -> latent -> AMF -> frame" << std::endl;
    std::cout << "=======================================\n" << std::endl;

    std::string prompt = "Personaje saltando, camara rotando 45 grados";
    if (argc > 1) {
        prompt = argv[1];
    }

    AMFWorldModelRuntime amf_runtime;
    amf_runtime.load_warm_export("../data/phase10a_warm_amf.json");
    amf_runtime.freeze_identity_from_geometry(2048, 1.0f);

    ExternalActionOrchestrator orchestrator;

    NeverLatentState state = make_initial_state();

    constexpr int width = 256;
    constexpr int height = 256;
    uint8_t* d_frame = nullptr;
    cudaMalloc(&d_frame, width * height * 4 * sizeof(uint8_t));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    for (int frame = 0; frame < 8; ++frame) {
        NeverActionVector action = orchestrator.prompt_to_action(prompt, frame);
        NeverLatentState predicted = amf_runtime.predict_host(state, action);
        NeverLatentState real_next = simulate_real_never_latent(state, action, frame);
        AMFOnlineLearnResult online = amf_runtime.learn_from_real_latent(state, action, real_next);
        state = real_next;
        amf_decode_latent_to_rgba_kernel<<<grid, block>>>(state, d_frame, width, height);
        cudaDeviceSynchronize();
        std::cout << "[AMF Loop] frame=" << frame
                  << " action=(" << action.force_x << ", " << action.force_y << ")"
                  << " predicted=(" << predicted.x << ", " << predicted.y << ", "
                  << predicted.vx << ", " << predicted.vy << ")"
                  << " latent=(" << state.x << ", " << state.y << ", "
                  << state.vx << ", " << state.vy << ")"
                  << " online_status=" << online.status
                  << " mse=" << online.mse_before << "->" << online.mse_after
                  << " online_cells=" << online.online_cells
                  << " identity_frozen=" << amf_runtime.identity().identity_frozen
                  << std::endl;
    }

    NeverLatentState probe_state = make_initial_state();
    probe_state.x = -0.72f;
    probe_state.y = 0.44f;
    probe_state.vx = -0.31f;
    probe_state.vy = 0.18f;
    NeverActionVector probe_action{0.73f, -0.64f, 37.0f, 1.0f};
    NeverLatentState probe_next = make_probe_next(probe_state, probe_action);
    NeverLatentState probe_before = amf_runtime.predict_host(probe_state, probe_action);
    float before_mse = 0.25f * (
        (probe_before.x - probe_next.x) * (probe_before.x - probe_next.x) +
        (probe_before.y - probe_next.y) * (probe_before.y - probe_next.y) +
        (probe_before.vx - probe_next.vx) * (probe_before.vx - probe_next.vx) +
        (probe_before.vy - probe_next.vy) * (probe_before.vy - probe_next.vy)
    );
    AMFOnlineLearnResult probe_result;
    for (int i = 0; i < 4; ++i) {
        probe_result = amf_runtime.learn_from_real_latent(probe_state, probe_action, probe_next);
    }
    std::cout << "[AMF Online Probe] score="
              << amf_runtime.online_probe_score(before_mse) << "->"
              << amf_runtime.online_probe_score(probe_result.mse_after)
              << " mse=" << before_mse << "->" << probe_result.mse_after
              << " final_status=" << probe_result.status
              << " online_cells=" << amf_runtime.online_cell_count()
              << std::endl;

    cudaFree(d_frame);
    std::cout << "\nN.E.V.E.R. Phase 10c AMF core ready." << std::endl;
    return 0;
}

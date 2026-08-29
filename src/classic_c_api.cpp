#include "gravity_lab/classic_c_api.h"

#include "gravity_lab/classic_environment.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

struct gdc_env {
    explicit gdc_env(gravity_lab::classic::Config config, std::filesystem::path path)
        : value(config, std::move(path)) {}
    gravity_lab::classic::Environment value;
    std::string track_name_cache;
};

namespace {
thread_local std::string last_error;

template <typename Function>
int guard(Function&& function) {
    try {
        function();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "unknown C++ exception";
        return -1;
    }
}

void copy_observation(const gravity_lab::classic::Observation& source, double* destination) {
    std::copy(source.begin(), source.end(), destination);
}
}  // namespace

extern "C" {

gdc_config gdc_default_config(void) {
    const gravity_lab::classic::Config config;
    return {config.level_group, config.track, config.league, config.frame_skip,
            config.max_episode_steps, config.seed};
}

gdc_env* gdc_create(const char* level_pack_path, gdc_config config) {
    gdc_env* result = nullptr;
    guard([&] {
        const gravity_lab::classic::Config native{config.level_group, config.track, config.league,
                                                   config.frame_skip, config.max_episode_steps, config.seed};
        const std::filesystem::path path = level_pack_path ? level_pack_path : "";
        result = new gdc_env(native, path);
    });
    return result;
}

void gdc_destroy(gdc_env* env) { delete env; }

int gdc_reset(gdc_env* env, uint64_t seed, double observation[GDC_OBSERVATION_SIZE]) {
    return guard([&] {
        if (!env || !observation) throw std::invalid_argument("gdc_reset received null pointer");
        copy_observation(env->value.reset(seed), observation);
    });
}

int gdc_step(gdc_env* env, int action, gdc_step_result* result) {
    return guard([&] {
        if (!env || !result) throw std::invalid_argument("gdc_step received null pointer");
        if (action < 0 || action >= gravity_lab::classic::kActionCount) {
            throw std::invalid_argument("classic action is outside [0, 8]");
        }
        const auto native = env->value.step(static_cast<gravity_lab::classic::Action>(action));
        copy_observation(native.observation, result->observation);
        result->reward = native.reward;
        result->terminated = native.terminated;
        result->truncated = native.truncated;
        result->finished = native.finished;
        result->crashed = native.crashed;
        result->wheelie_finish = native.wheelie_finish;
        result->physics_code = native.physics_code;
    });
}

const char* gdc_track_name(gdc_env* env) {
    if (!env) {
        last_error = "gdc_track_name received null pointer";
        return nullptr;
    }
    try {
        env->track_name_cache = env->value.track_name();
        last_error.clear();
        return env->track_name_cache.c_str();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

int gdc_track_count(gdc_env* env, uint32_t level_group, uint32_t* count) {
    return guard([&] {
        if (!env || !count) throw std::invalid_argument("gdc_track_count received null pointer");
        *count = env->value.track_count(level_group);
    });
}

const char* gdc_last_error(void) { return last_error.c_str(); }

}  // extern "C"

#include "gravity_lab/c_api.h"

#include "gravity_lab/environment.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>

struct gd_env {
    explicit gd_env(gravity_lab::Environment value) : value(std::move(value)) {}
    gravity_lab::Environment value;
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

void copy_observation(const gravity_lab::Observation& source, double* destination) {
    std::copy(source.begin(), source.end(), destination);
}
}  // namespace

extern "C" {

gd_config gd_default_config(void) {
    const gravity_lab::Config config;
    return {config.time_step, config.frame_skip, config.max_episode_steps, config.seed};
}

gd_env* gd_create(const char* map_path, gd_config config) {
    gd_env* result = nullptr;
    guard([&] {
        if (!map_path) throw std::invalid_argument("map_path cannot be null");
        gravity_lab::Config native{config.time_step, config.frame_skip, config.max_episode_steps, config.seed};
        result = new gd_env(gravity_lab::Environment(gravity_lab::Map::load(map_path), native));
    });
    return result;
}

void gd_destroy(gd_env* env) { delete env; }

int gd_reset(gd_env* env, uint64_t seed, double observation[GD_OBSERVATION_SIZE]) {
    return guard([&] {
        if (!env || !observation) throw std::invalid_argument("gd_reset received null pointer");
        copy_observation(env->value.reset(seed), observation);
    });
}

int gd_step(gd_env* env, int action, gd_step_result* result) {
    return guard([&] {
        if (!env || !result) throw std::invalid_argument("gd_step received null pointer");
        if (action < 0 || action >= gravity_lab::kActionCount) {
            throw std::invalid_argument("action is outside [0, 8]");
        }
        const auto native = env->value.step(static_cast<gravity_lab::Action>(action));
        copy_observation(native.observation, result->observation);
        result->reward = native.reward;
        result->terminated = native.terminated;
        result->truncated = native.truncated;
        result->finished = native.finished;
        result->crashed = native.crashed;
    });
}

const char* gd_last_error(void) { return last_error.c_str(); }

}  // extern "C"

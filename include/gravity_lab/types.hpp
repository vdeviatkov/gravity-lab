#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gravity_lab {

constexpr std::size_t kObservationSize = 12;
using Observation = std::array<double, kObservationSize>;

enum class Action : std::int32_t {
    Coast = 0,
    Throttle = 1,
    Brake = 2,
    LeanBack = 3,
    LeanForward = 4,
    ThrottleLeanBack = 5,
    ThrottleLeanForward = 6,
    BrakeLeanBack = 7,
    BrakeLeanForward = 8,
};

constexpr std::int32_t kActionCount = 9;

struct Config {
    double time_step{1.0 / 120.0};
    std::uint32_t frame_skip{4};
    std::uint32_t max_episode_steps{3'000};
    std::uint64_t seed{1};
};

struct State {
    double x{};
    double y{};
    double velocity_x{};
    double velocity_y{};
    double angle{};
    double angular_velocity{};
    std::uint32_t episode_step{};
    bool grounded{};
};

struct StepResult {
    Observation observation{};
    double reward{};
    bool terminated{};
    bool truncated{};
    bool finished{};
    bool crashed{};
};

}  // namespace gravity_lab

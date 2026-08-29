#include "gravity_lab/environment.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gravity_lab {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRideHeight = 0.48;

double wrap_angle(double value) {
    return std::remainder(value, 2.0 * kPi);
}

struct Controls {
    double drive;
    double lean;
};

Controls decode(Action action) {
    switch (action) {
        case Action::Coast: return {0.0, 0.0};
        case Action::Throttle: return {1.0, 0.0};
        case Action::Brake: return {-1.0, 0.0};
        case Action::LeanBack: return {0.0, 1.0};
        case Action::LeanForward: return {0.0, -1.0};
        case Action::ThrottleLeanBack: return {1.0, 1.0};
        case Action::ThrottleLeanForward: return {1.0, -1.0};
        case Action::BrakeLeanBack: return {-1.0, 1.0};
        case Action::BrakeLeanForward: return {-1.0, -1.0};
    }
    throw std::invalid_argument("action is outside [0, 8]");
}

}  // namespace

Environment::Environment(Map map, Config config)
    : map_(std::move(map)), config_(config), seed_(config.seed) {
    if (!(config_.time_step > 0.0 && config_.time_step <= 0.1)) {
        throw std::invalid_argument("time_step must be in (0, 0.1]");
    }
    if (config_.frame_skip == 0 || config_.frame_skip > 100) {
        throw std::invalid_argument("frame_skip must be in [1, 100]");
    }
    if (config_.max_episode_steps == 0) {
        throw std::invalid_argument("max_episode_steps must be positive");
    }
    reset();
}

Observation Environment::reset() {
    return reset(seed_);
}

Observation Environment::reset(std::uint64_t seed) {
    seed_ = seed;
    state_ = {};
    state_.x = map_.start_x();
    state_.y = map_.height_at(state_.x) + kRideHeight;
    state_.angle = map_.slope_at(state_.x);
    state_.grounded = true;
    terminated_ = truncated_ = finished_ = crashed_ = false;
    return observation();
}

void Environment::integrate(Action action) {
    const Controls controls = decode(action);
    const double dt = config_.time_step;
    const double ground = map_.height_at(state_.x);
    const bool near_ground = state_.y <= ground + kRideHeight + 0.08;

    state_.velocity_y -= 9.81 * dt;
    state_.angular_velocity += controls.lean * 4.2 * dt;
    if (near_ground) {
        const double slope = map_.slope_at(state_.x);
        state_.velocity_x += controls.drive * 7.0 * std::cos(slope) * dt;
        state_.velocity_x += -9.81 * std::sin(slope) * 0.35 * dt;
        state_.velocity_x *= std::pow(0.994, dt * 120.0);
        const double angle_error = wrap_angle(slope - state_.angle);
        state_.angular_velocity += angle_error * 18.0 * dt;
        state_.angular_velocity *= std::pow(0.90, dt * 120.0);
    } else {
        state_.velocity_x += controls.drive * 1.2 * dt;
        state_.angular_velocity *= std::pow(0.998, dt * 120.0);
    }

    state_.velocity_x = std::clamp(state_.velocity_x, -7.0, 14.0);
    state_.angular_velocity = std::clamp(state_.angular_velocity, -8.0, 8.0);
    state_.x += state_.velocity_x * dt;
    state_.y += state_.velocity_y * dt;
    state_.angle = wrap_angle(state_.angle + state_.angular_velocity * dt);

    const double new_ground = map_.height_at(state_.x);
    if (state_.y <= new_ground + kRideHeight) {
        const double impact = -state_.velocity_y;
        const double slope = map_.slope_at(state_.x);
        state_.y = new_ground + kRideHeight;
        state_.velocity_y = std::max(0.0, state_.velocity_y * -0.05);
        state_.grounded = true;
        const double angle_error = std::abs(wrap_angle(state_.angle - slope));
        if ((angle_error > 1.42 && std::abs(state_.velocity_x) > 0.8) || impact > 8.0) {
            crashed_ = terminated_ = true;
        }
    } else {
        state_.grounded = false;
    }

    const double head_x = state_.x - 0.15 * std::sin(state_.angle);
    const double head_y = state_.y + 0.72 * std::cos(state_.angle);
    if (head_y <= map_.height_at(head_x) + 0.03 || state_.x < map_.points().front().x) {
        crashed_ = terminated_ = true;
    }
    if (state_.x >= map_.finish_x()) {
        finished_ = terminated_ = true;
    }
}

StepResult Environment::step(Action action) {
    if (done()) throw std::logic_error("step called after episode ended; call reset first");
    const double previous_x = state_.x;
    for (std::uint32_t i = 0; i < config_.frame_skip && !terminated_; ++i) integrate(action);
    ++state_.episode_step;
    if (!terminated_ && state_.episode_step >= config_.max_episode_steps) truncated_ = true;

    double reward = (state_.x - previous_x) * 0.1 - 0.001;
    if (finished_) reward += 10.0;
    if (crashed_) reward -= 5.0;
    return result(reward);
}

Observation Environment::observation() const {
    const double terrain = map_.height_at(state_.x);
    const double span = std::max(1.0, map_.finish_x() - map_.start_x());
    return {
        (state_.x - map_.start_x()) / span,
        (state_.y - terrain) / 5.0,
        state_.velocity_x / 14.0,
        state_.velocity_y / 10.0,
        std::sin(state_.angle),
        std::cos(state_.angle),
        state_.angular_velocity / 8.0,
        map_.slope_at(state_.x) / (kPi / 2.0),
        (map_.height_at(state_.x + 1.0) - terrain) / 5.0,
        (map_.height_at(state_.x + 3.0) - terrain) / 5.0,
        (map_.height_at(state_.x + 6.0) - terrain) / 5.0,
        state_.grounded ? 1.0 : 0.0,
    };
}

StepResult Environment::result(double reward) const {
    return {observation(), reward, terminated_, truncated_, finished_, crashed_};
}

}  // namespace gravity_lab

#include "gravity_lab/environment.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void near(double actual, double expected, double tolerance, const std::string& message) {
    check(std::abs(actual - expected) <= tolerance,
          message + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) + ")");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("test requires map path");
        const std::filesystem::path path = argv[1];
        const auto map = gravity_lab::Map::load(path);
        check(map.name() == "training", "map name is parsed");
        near(map.height_at(7.0), 0.6, 1e-12, "terrain is linearly interpolated");
        check(map.slope_at(7.0) > 0.0, "uphill slope is positive");

        gravity_lab::Config config;
        config.seed = 42;
        config.max_episode_steps = 3;
        gravity_lab::Environment first(map, config);
        gravity_lab::Environment second(map, config);
        check(first.reset(99) == second.reset(99), "reset observation is deterministic");
        for (int i = 0; i < 3; ++i) {
            const auto a = first.step(gravity_lab::Action::ThrottleLeanBack);
            const auto b = second.step(gravity_lab::Action::ThrottleLeanBack);
            check(a.observation == b.observation, "equal actions produce equal observations");
            near(a.reward, b.reward, 0.0, "equal actions produce equal rewards");
        }
        check(first.done(), "episode ends at configured step limit");

        gravity_lab::Config movement_config;
        movement_config.max_episode_steps = 200;
        gravity_lab::Environment moving(map, movement_config);
        const double initial_x = moving.state().x;
        for (int i = 0; i < 10 && !moving.done(); ++i) moving.step(gravity_lab::Action::Throttle);
        check(moving.state().x > initial_x, "throttle moves bike forward");
        check(moving.observation().size() == gravity_lab::kObservationSize, "observation shape is stable");

        bool rejected_step_after_done = false;
        try {
            first.step(gravity_lab::Action::Coast);
        } catch (const std::logic_error&) {
            rejected_step_after_done = true;
        }
        check(rejected_step_after_done, "step after done is rejected");

        bool rejected_config = false;
        try {
            gravity_lab::Config bad;
            bad.frame_skip = 0;
            gravity_lab::Environment invalid(map, bad);
        } catch (const std::invalid_argument&) {
            rejected_config = true;
        }
        check(rejected_config, "invalid simulation configuration is rejected");

        if (failures == 0) std::cout << "all C++ environment tests passed\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "test setup failed: " << error.what() << '\n';
        return 2;
    }
}

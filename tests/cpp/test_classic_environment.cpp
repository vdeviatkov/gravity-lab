#include "gravity_lab/classic_environment.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct Transition {
    gravity_lab::classic::Observation observation;
    double reward;
    int code;
    bool operator==(const Transition&) const = default;
};

std::vector<Transition> trajectory(
    gravity_lab::classic::Config config,
    const std::filesystem::path& level_pack = {}
) {
    gravity_lab::classic::Environment env(config, level_pack);
    std::vector<Transition> result;
    env.reset(123);
    const std::vector<gravity_lab::classic::Action> actions{
        gravity_lab::classic::Action::Throttle,
        gravity_lab::classic::Action::ThrottleLeanBack,
        gravity_lab::classic::Action::ThrottleLeanForward,
        gravity_lab::classic::Action::Coast,
    };
    for (const auto action : actions) {
        const auto step = env.step(action);
        result.push_back({step.observation, step.reward, step.physics_code});
        if (env.done()) break;
    }
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        gravity_lab::classic::Config config;
        config.max_episode_steps = 10;
        const auto first = trajectory(config);
        const auto second = trajectory(config);
        check(first.size() == second.size(), "deterministic trajectories have equal length");
        check(first == second, "fixed actions produce bitwise-equal classic trajectories");
        if (argc > 1) {
            check(first == trajectory(config, argv[1]),
                  "embedded and external copies of the same level pack agree");
        }

        {
            gravity_lab::classic::Environment env(config);
            const auto initial = env.reset(999);
            check(initial.size() == gravity_lab::classic::kObservationSize, "classic observation size is stable");
            for (double value : initial) check(std::isfinite(value), "classic observations are finite");
            check(!env.track_name().empty(), "classic track name is exposed");
            check(env.track_count(0) > 0, "classic track count is exposed");

            bool singleton_rejected = false;
            try {
                gravity_lab::classic::Environment second_env(config);
            } catch (const std::runtime_error&) {
                singleton_rejected = true;
            }
            check(singleton_rejected, "unsafe concurrent classic environment is rejected");
        }

        gravity_lab::classic::Config short_config;
        short_config.max_episode_steps = 2;
        {
            gravity_lab::classic::Environment env(short_config);
            env.step(gravity_lab::classic::Action::Coast);
            const auto final = env.step(gravity_lab::classic::Action::Coast);
            check(final.truncated && !final.terminated, "classic time limit is distinct from termination");
            bool rejected = false;
            try { env.step(gravity_lab::classic::Action::Coast); }
            catch (const std::logic_error&) { rejected = true; }
            check(rejected, "classic step after done is rejected");
        }

        {
            auto finish_config = config;
            finish_config.max_episode_steps = 300;
            gravity_lab::classic::Environment env(finish_config);
            gravity_lab::classic::StepResult final;
            while (!env.done()) final = env.step(gravity_lab::classic::Action::Throttle);
            check(final.terminated && final.finished && !final.crashed && !final.truncated,
                  "throttle baseline reaches the Intro finish as a terminal transition");
            check(final.reward > 9.0, "finish transition includes its reward bonus");
        }

        {
            auto crash_config = config;
            crash_config.max_episode_steps = 1'000;
            gravity_lab::classic::Environment env(crash_config);
            gravity_lab::classic::StepResult final;
            while (!env.done()) final = env.step(gravity_lab::classic::Action::ThrottleLeanBack);
            check(final.terminated && final.crashed && !final.finished && !final.truncated,
                  "crash is exposed as a terminal transition");
            check(final.reward < -4.0, "crash transition includes its reward penalty");
        }

        bool bad_track_rejected = false;
        try {
            auto bad = config;
            bad.track = 1'000'000;
            gravity_lab::classic::Environment env(bad);
        } catch (const std::invalid_argument&) {
            bad_track_rejected = true;
        }
        check(bad_track_rejected, "invalid classic track is rejected");

        if (failures == 0) std::cout << "all classic environment tests passed\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "test setup failed: " << error.what() << '\n';
        return 2;
    }
}

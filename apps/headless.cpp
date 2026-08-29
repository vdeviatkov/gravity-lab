#include "gravity_lab/environment.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::filesystem::path map{"maps/training.gdmap"};
    std::uint64_t seed{1};
    std::uint32_t episodes{3};
    double time_step{1.0 / 120.0};
    std::uint32_t frame_skip{4};
    std::uint32_t max_steps{3'000};
    bool heuristic{false};
};

template <typename T>
T number(std::string_view value, std::string_view option) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::runtime_error("invalid value for " + std::string(option));
    }
    return result;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argc) throw std::runtime_error("missing value after " + std::string(arg));
            return argv[i];
        };
        if (arg == "--map") options.map = next();
        else if (arg == "--seed") options.seed = number<std::uint64_t>(next(), arg);
        else if (arg == "--episodes") options.episodes = number<std::uint32_t>(next(), arg);
        else if (arg == "--frame-skip") options.frame_skip = number<std::uint32_t>(next(), arg);
        else if (arg == "--max-steps") options.max_steps = number<std::uint32_t>(next(), arg);
        else if (arg == "--dt") options.time_step = number<double>(next(), arg);
        else if (arg == "--policy") {
            const auto policy = next();
            if (policy == "heuristic") options.heuristic = true;
            else if (policy != "random") throw std::runtime_error("policy must be random or heuristic");
        } else if (arg == "--help") {
            std::cout << "Usage: gravity_lab_headless [--map PATH] [--seed N] [--episodes N]\n"
                         "       [--dt SECONDS] [--frame-skip N] [--max-steps N]\n"
                         "       [--policy random|heuristic]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + std::string(arg));
    }
    return options;
}

gravity_lab::Action heuristic(const gravity_lab::Environment& env) {
    const auto& state = env.state();
    const double slope_ahead = env.map().slope_at(state.x + 1.5);
    if (!state.grounded) {
        if (state.angle > slope_ahead + 0.12) return gravity_lab::Action::ThrottleLeanForward;
        if (state.angle < slope_ahead - 0.12) return gravity_lab::Action::ThrottleLeanBack;
    }
    return gravity_lab::Action::Throttle;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        gravity_lab::Config config{options.time_step, options.frame_skip, options.max_steps, options.seed};
        gravity_lab::Environment env(gravity_lab::Map::load(options.map), config);
        std::mt19937_64 exploration(options.seed + 10'000);
        std::uniform_int_distribution<int> actions(0, gravity_lab::kActionCount - 1);

        std::cout << "map,episode,seed,reward,steps,progress,finished,crashed,truncated\n";
        for (std::uint32_t episode = 0; episode < options.episodes; ++episode) {
            const auto episode_seed = options.seed + episode;
            env.reset(episode_seed);
            double total_reward = 0.0;
            while (!env.done()) {
                const auto action = options.heuristic
                    ? heuristic(env)
                    : static_cast<gravity_lab::Action>(actions(exploration));
                total_reward += env.step(action).reward;
            }
            const auto& state = env.state();
            const double progress = (state.x - env.map().start_x()) /
                                    (env.map().finish_x() - env.map().start_x());
            std::cout << env.map().name() << ',' << episode << ',' << episode_seed << ','
                      << total_reward << ',' << state.episode_step << ',' << progress << ','
                      << env.finished() << ',' << env.crashed() << ',' << env.truncated() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

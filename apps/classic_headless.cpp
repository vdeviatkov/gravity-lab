#include "gravity_lab/classic_environment.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    gravity_lab::classic::Config config;
    std::filesystem::path level_pack;
    std::uint32_t episodes{3};
    bool random_policy{false};
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
        if (arg == "--group") options.config.level_group = number<std::uint32_t>(next(), arg);
        else if (arg == "--level-pack") options.level_pack = std::string(next());
        else if (arg == "--track") options.config.track = number<std::uint32_t>(next(), arg);
        else if (arg == "--league") options.config.league = number<std::uint32_t>(next(), arg);
        else if (arg == "--frame-skip") options.config.frame_skip = number<std::uint32_t>(next(), arg);
        else if (arg == "--max-steps") options.config.max_episode_steps = number<std::uint32_t>(next(), arg);
        else if (arg == "--seed") options.config.seed = number<std::uint64_t>(next(), arg);
        else if (arg == "--episodes") options.episodes = number<std::uint32_t>(next(), arg);
        else if (arg == "--policy") {
            const auto policy = next();
            if (policy == "random") options.random_policy = true;
            else if (policy != "throttle") throw std::runtime_error("policy must be random or throttle");
        } else if (arg == "--help") {
            std::cout << "Usage: gravity_lab_classic_headless [--level-pack FILE] [--group 0..2]\n"
                         "       [--track N] [--league 0..3]\n"
                         "       [--frame-skip N] [--max-steps N] [--episodes N] [--seed N]\n"
                         "       [--policy random|throttle]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + std::string(arg));
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        gravity_lab::classic::Environment env(options.config, options.level_pack);
        std::mt19937_64 exploration(options.config.seed + 10'000);
        std::uniform_int_distribution<int> actions(0, gravity_lab::classic::kActionCount - 1);
        std::cout << "environment,track,episode,seed,reward,steps,progress,finished,crashed,truncated,wheelie\n";
        for (std::uint32_t episode = 0; episode < options.episodes; ++episode) {
            const auto seed = options.config.seed + episode;
            auto observation = env.reset(seed);
            double total_reward = 0.0;
            gravity_lab::classic::StepResult result;
            while (!env.done()) {
                const auto action = options.random_policy
                    ? static_cast<gravity_lab::classic::Action>(actions(exploration))
                    : gravity_lab::classic::Action::Throttle;
                result = env.step(action);
                observation = result.observation;
                total_reward += result.reward;
            }
            std::cout << "classic-v1," << env.track_name() << ',' << episode << ',' << seed << ','
                      << total_reward << ',' << env.episode_step() << ',' << observation[0] << ','
                      << result.finished << ',' << result.crashed << ',' << result.truncated << ','
                      << result.wheelie_finish << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

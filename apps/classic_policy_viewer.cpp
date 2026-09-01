#include "gravity_lab/classic_environment.hpp"
#include "gravity_lab/classic_renderer.hpp"
#include "gravity_lab/dense_policy.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

struct Options {
    gravity_lab::classic::Config config;
    std::filesystem::path level_pack;
    std::filesystem::path policy;
    std::uint32_t episodes{3};
    std::uint32_t hold_milliseconds{1'000};
    double fps{30.0};
    bool validate_only{false};
};

template <typename T>
T integer(std::string_view value, std::string_view option) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::runtime_error("invalid value for " + std::string(option));
    }
    return result;
}

double real(std::string_view value, std::string_view option) {
    std::size_t used = 0;
    const double result = std::stod(std::string(value), &used);
    if (used != value.size() || !std::isfinite(result)) {
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
        if (arg == "--policy") options.policy = std::string(next());
        else if (arg == "--level-pack") options.level_pack = std::string(next());
        else if (arg == "--group") options.config.level_group = integer<std::uint32_t>(next(), arg);
        else if (arg == "--track") options.config.track = integer<std::uint32_t>(next(), arg);
        else if (arg == "--league") options.config.league = integer<std::uint32_t>(next(), arg);
        else if (arg == "--frame-skip") options.config.frame_skip = integer<std::uint32_t>(next(), arg);
        else if (arg == "--max-steps") options.config.max_episode_steps = integer<std::uint32_t>(next(), arg);
        else if (arg == "--episodes") options.episodes = integer<std::uint32_t>(next(), arg);
        else if (arg == "--seed") options.config.seed = integer<std::uint64_t>(next(), arg);
        else if (arg == "--fps") options.fps = real(next(), arg);
        else if (arg == "--hold-ms") options.hold_milliseconds = integer<std::uint32_t>(next(), arg);
        else if (arg == "--validate-only") options.validate_only = true;
        else if (arg == "--help") {
            std::cout << "Usage: gravity_lab_classic_viewer --policy FILE [options]\n"
                         "  --level-pack FILE  custom .mrg level pack\n"
                         "  --group N --track N --league N\n"
                         "  --frame-skip N --max-steps N --episodes N --seed N\n"
                         "  --fps N --hold-ms N --validate-only\n"
                         "Escape or the window close button stops playback.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + std::string(arg));
        }
    }
    if (options.policy.empty()) throw std::runtime_error("--policy is required");
    if (options.episodes == 0) throw std::runtime_error("episodes must be positive");
    if (options.fps < 0.0 || options.fps > 1'000.0) throw std::runtime_error("fps must be in [0, 1000]");
    return options;
}

void validate(const gravity_lab::DenseQPolicy& policy) {
    if (policy.environment_id() != "gravity-lab-classic-v1") {
        throw std::runtime_error("policy environment must be gravity-lab-classic-v1");
    }
    // A policy trained before the obstacle-ray sensor (or before acceleration) was added
    // declares a smaller observation_size; every prefix length in
    // [kBaseObservationSize, kObservationSize] is a compatible, unchanged prefix of the current
    // vector (see classic_environment.hpp).
    if (policy.observation_size() < gravity_lab::classic::kBaseObservationSize ||
        policy.observation_size() > gravity_lab::classic::kObservationSize) {
        throw std::runtime_error("policy observation dimension does not match classic-v1");
    }
    if (policy.action_count() != static_cast<std::size_t>(gravity_lab::classic::kActionCount)) {
        throw std::runtime_error("policy action dimension does not match classic-v1");
    }
}

// The number of obstacle rays to compute so a policy's own observation region is populated with
// real values: derived from its declared observation_size (see classic_environment.hpp for the
// region layout), clamped to a valid ray count regardless of whether the policy uses any rays.
std::uint32_t obstacle_ray_count_for(const gravity_lab::DenseQPolicy& policy) {
    const auto base = gravity_lab::classic::kBaseObservationSize;
    const auto max_rays = gravity_lab::classic::kMaxObstacleRayCount;
    const std::size_t requested = policy.observation_size() > base ? policy.observation_size() - base : 0;
    return static_cast<std::uint32_t>(std::clamp<std::size_t>(requested, 1, max_rays));
}

class FramePacer {
public:
    explicit FramePacer(double fps) : enabled_(fps > 0.0), next_(std::chrono::steady_clock::now()) {
        if (enabled_) {
            duration_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / fps));
        }
    }

    void wait() {
        if (!enabled_) return;
        next_ += duration_;
        std::this_thread::sleep_until(next_);
    }

private:
    bool enabled_;
    std::chrono::steady_clock::time_point next_;
    std::chrono::steady_clock::duration duration_{};
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const auto policy = gravity_lab::DenseQPolicy::load(options.policy);
        validate(policy);
        if (options.validate_only) {
            std::cout << "valid policy environment=" << policy.environment_id()
                      << " observations=" << policy.observation_size()
                      << " actions=" << policy.action_count()
                      << " layers=" << policy.layers().size() << '\n';
            return 0;
        }

        auto config = options.config;
        config.obstacle_ray_count = obstacle_ray_count_for(policy);
        gravity_lab::classic::Environment environment(config, options.level_pack);
        gravity_lab::classic::Renderer renderer(
            environment, "Gravity Lab - " + environment.track_name() + " - learned policy");
        FramePacer pacer(options.fps);
        for (std::uint32_t episode = 0; episode < options.episodes && renderer.open(); ++episode) {
            auto observation = environment.reset(options.config.seed + episode);
            std::uint64_t elapsed_milliseconds = 0;
            double total_reward = 0.0;
            gravity_lab::classic::StepResult result;
            renderer.show_message(environment.track_name(), 1'000);
            renderer.render_frame(elapsed_milliseconds);
            while (!environment.done() && renderer.open()) {
                const auto action = policy.action(
                    std::span<const double>(observation.data(), policy.observation_size()));
                result = environment.step(static_cast<gravity_lab::classic::Action>(action));
                observation = result.observation;
                total_reward += result.reward;
                elapsed_milliseconds += 20ULL * options.config.frame_skip;
                renderer.render_frame(elapsed_milliseconds);
                pacer.wait();
            }
            if (!renderer.open()) break;
            const std::string outcome = result.finished ? (result.wheelie_finish ? "Wheelie!" : "Finished")
                : result.crashed ? "Crashed" : "Time limit";
            renderer.show_message(outcome, options.hold_milliseconds);
            const auto hold_until = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(options.hold_milliseconds);
            while (renderer.open() && std::chrono::steady_clock::now() < hold_until) {
                renderer.render_frame(elapsed_milliseconds);
                if (options.fps > 0.0) pacer.wait();
                else std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cout << "episode=" << episode << " seed=" << options.config.seed + episode
                      << " reward=" << total_reward << " steps=" << environment.episode_step()
                      << " progress=" << observation[0] << " finished=" << result.finished
                      << " crashed=" << result.crashed << " truncated=" << result.truncated << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

#include "gravity_lab/classic_environment.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using State = std::array<int, 6>;
using Values = std::array<double, gravity_lab::classic::kActionCount>;

struct StateHash {
    std::size_t operator()(const State& state) const noexcept {
        std::size_t result = 0xcbf29ce484222325ULL;
        for (const int value : state) {
            result ^= static_cast<std::size_t>(value + 257);
            result *= 0x100000001b3ULL;
        }
        return result;
    }
};

using Table = std::unordered_map<State, Values, StateHash>;

struct Options {
    gravity_lab::classic::Config config;
    std::filesystem::path level_pack;
    std::uint32_t episodes{2'000};
    std::uint32_t eval_episodes{20};
    std::uint64_t eval_seed{1'000'001};
    double alpha{0.15};
    double gamma{0.99};
    std::filesystem::path checkpoint{"artifacts/classic_q.tsv"};
    bool eval_only{false};
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
    if (used != value.size() || !std::isfinite(result)) throw std::runtime_error("invalid value for " + std::string(option));
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
        if (arg == "--group") options.config.level_group = integer<std::uint32_t>(next(), arg);
        else if (arg == "--level-pack") options.level_pack = std::string(next());
        else if (arg == "--track") options.config.track = integer<std::uint32_t>(next(), arg);
        else if (arg == "--league") options.config.league = integer<std::uint32_t>(next(), arg);
        else if (arg == "--frame-skip") options.config.frame_skip = integer<std::uint32_t>(next(), arg);
        else if (arg == "--max-steps") options.config.max_episode_steps = integer<std::uint32_t>(next(), arg);
        else if (arg == "--episodes") options.episodes = integer<std::uint32_t>(next(), arg);
        else if (arg == "--eval-episodes") options.eval_episodes = integer<std::uint32_t>(next(), arg);
        else if (arg == "--train-seed") options.config.seed = integer<std::uint64_t>(next(), arg);
        else if (arg == "--eval-seed") options.eval_seed = integer<std::uint64_t>(next(), arg);
        else if (arg == "--alpha") options.alpha = real(next(), arg);
        else if (arg == "--gamma") options.gamma = real(next(), arg);
        else if (arg == "--checkpoint") options.checkpoint = next();
        else if (arg == "--eval-only") options.eval_only = true;
        else if (arg == "--help") {
            std::cout << "Usage: gravity_lab_classic_q [--level-pack FILE] [--group N] [--track N] [--league N]\n"
                         "       [--episodes N] [--eval-episodes N] [--train-seed N] [--eval-seed N]\n"
                         "       [--frame-skip N] [--max-steps N] [--alpha X] [--gamma X]\n"
                         "       [--checkpoint PATH] [--eval-only]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + std::string(arg));
    }
    if (!options.eval_only && options.episodes == 0) throw std::runtime_error("episodes must be positive");
    if (options.eval_episodes == 0) throw std::runtime_error("eval_episodes must be positive");
    if (!(options.alpha > 0.0 && options.alpha <= 1.0)) throw std::runtime_error("alpha must be in (0, 1]");
    if (!(options.gamma >= 0.0 && options.gamma <= 1.0)) throw std::runtime_error("gamma must be in [0, 1]");
    return options;
}

int bucket(double value, double scale) {
    return static_cast<int>(std::lround(std::clamp(value, -3.0, 3.0) * scale));
}

State encode(const gravity_lab::classic::Observation& observation) {
    return {bucket(observation[0], 30.0), bucket(observation[6], 12.0),
            bucket(observation[7], 12.0), bucket(observation[9], 30.0),
            bucket(observation[13], 30.0), bucket(observation[2], 1.0)};
}

std::size_t greedy(const Values& values) {
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

void save(const std::filesystem::path& path, const Table& table, const Options& options) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) throw std::runtime_error("cannot write checkpoint: " + temporary);
    output << "gravity-lab-classic-tabular-q-tsv-v1\n";
    output << std::setprecision(17);
    output << options.config.level_group << ' ' << options.config.track << ' ' << options.config.league << ' '
           << options.config.frame_skip << ' ' << options.config.max_episode_steps << '\n';
    output << "level_pack " << std::quoted(options.level_pack.empty()
        ? std::string("embedded") : std::filesystem::absolute(options.level_pack).lexically_normal().string()) << '\n';
    output << "training " << options.config.seed << ' ' << options.episodes << ' '
           << options.alpha << ' ' << options.gamma << '\n';
    output << "encoder classic-tabular-v1\n";
    std::vector<State> states;
    states.reserve(table.size());
    for (const auto& [state, values] : table) {
        (void)values;
        states.push_back(state);
    }
    std::sort(states.begin(), states.end());
    for (const auto& state : states) {
        const auto& values = table.at(state);
        for (const int value : state) output << value << ' ';
        for (const double value : values) output << value << ' ';
        output << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing checkpoint: " + temporary);
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
        if (error) throw std::runtime_error("cannot replace checkpoint: " + error.message());
    }
}

Table load(const std::filesystem::path& path, const Options& options) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read checkpoint: " + path.string());
    std::string format;
    std::getline(input, format);
    if (format != "gravity-lab-classic-tabular-q-tsv-v1") throw std::runtime_error("unsupported checkpoint format");
    gravity_lab::classic::Config stored;
    if (!(input >> stored.level_group >> stored.track >> stored.league >> stored.frame_skip
          >> stored.max_episode_steps)) {
        throw std::runtime_error("malformed checkpoint environment configuration");
    }
    if (stored.level_group != options.config.level_group || stored.track != options.config.track ||
        stored.league != options.config.league || stored.frame_skip != options.config.frame_skip ||
        stored.max_episode_steps != options.config.max_episode_steps) {
        throw std::runtime_error("checkpoint environment configuration does not match CLI configuration");
    }
    std::string label;
    std::string stored_level_pack;
    input >> label >> std::quoted(stored_level_pack);
    const std::string selected_level_pack = options.level_pack.empty()
        ? "embedded" : std::filesystem::absolute(options.level_pack).lexically_normal().string();
    if (label != "level_pack" || stored_level_pack != selected_level_pack) {
        throw std::runtime_error("checkpoint level pack does not match CLI configuration");
    }
    std::uint64_t stored_seed{};
    std::uint32_t stored_episodes{};
    double stored_alpha{};
    double stored_gamma{};
    if (!(input >> label >> stored_seed >> stored_episodes >> stored_alpha >> stored_gamma) || label != "training") {
        throw std::runtime_error("malformed checkpoint training metadata");
    }
    std::string encoder;
    if (!(input >> label >> encoder) || label != "encoder" || encoder != "classic-tabular-v1") {
        throw std::runtime_error("unsupported checkpoint state encoder");
    }
    Table table;
    while (true) {
        State state{};
        Values values{};
        if (!(input >> state[0])) {
            if (input.eof()) break;
            throw std::runtime_error("malformed checkpoint Q table");
        }
        for (std::size_t i = 1; i < state.size(); ++i) {
            if (!(input >> state[i])) throw std::runtime_error("malformed checkpoint Q table");
        }
        for (double& value : values) {
            if (!(input >> value) || !std::isfinite(value)) {
                throw std::runtime_error("malformed checkpoint Q table");
            }
        }
        if (!table.emplace(state, values).second) throw std::runtime_error("duplicate checkpoint state");
    }
    return table;
}

void evaluate(gravity_lab::classic::Environment& env, const Table& table, const Options& options) {
    std::vector<double> rewards;
    std::vector<double> progresses;
    std::uint64_t total_steps = 0;
    std::uint32_t finishes = 0;
    std::uint32_t crashes = 0;
    const Values zeros{};
    for (std::uint32_t episode = 0; episode < options.eval_episodes; ++episode) {
        auto observation = env.reset(options.eval_seed + episode);
        double reward = 0.0;
        gravity_lab::classic::StepResult result;
        do {
            const auto found = table.find(encode(observation));
            const auto action = greedy(found == table.end() ? zeros : found->second);
            result = env.step(static_cast<gravity_lab::classic::Action>(action));
            observation = result.observation;
            reward += result.reward;
            ++total_steps;
        } while (!env.done());
        rewards.push_back(reward);
        progresses.push_back(observation[0]);
        finishes += result.finished;
        crashes += result.crashed;
    }
    auto sorted = rewards;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 ? sorted[sorted.size() / 2]
        : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
    const double mean_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0) / rewards.size();
    const double mean_progress = std::accumulate(progresses.begin(), progresses.end(), 0.0) / progresses.size();
    std::cout << "evaluation_episodes=" << options.eval_episodes << " mean_reward=" << mean_reward
              << " median_reward=" << median << " mean_length="
              << static_cast<double>(total_steps) / options.eval_episodes << " mean_progress=" << mean_progress
              << " finish_rate=" << static_cast<double>(finishes) / options.eval_episodes
              << " crash_rate=" << static_cast<double>(crashes) / options.eval_episodes
              << " exploration=0\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        Table table = options.eval_only ? load(options.checkpoint, options) : Table{};
        gravity_lab::classic::Environment env(options.config, options.level_pack);
        std::mt19937_64 exploration(options.config.seed + 10'000);
        std::uniform_int_distribution<std::size_t> actions(0, gravity_lab::classic::kActionCount - 1);
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        if (!options.eval_only) {
            for (std::uint32_t episode = 0; episode < options.episodes; ++episode) {
                auto observation = env.reset(options.config.seed + episode);
                State state = encode(observation);
                const double epsilon = 0.05 + 0.95 * std::exp(
                    -static_cast<double>(episode) / std::max(1.0, options.episodes * 0.30));
                double total_reward = 0.0;
                gravity_lab::classic::StepResult result;
                do {
                    Values& values = table[state];
                    const std::size_t action = chance(exploration) < epsilon ? actions(exploration) : greedy(values);
                    result = env.step(static_cast<gravity_lab::classic::Action>(action));
                    const State next_state = encode(result.observation);
                    const double bootstrap = result.terminated ? 0.0 : options.gamma * *std::max_element(table[next_state].begin(), table[next_state].end());
                    values[action] += options.alpha * (result.reward + bootstrap - values[action]);
                    total_reward += result.reward;
                    state = next_state;
                } while (!env.done());
                if (episode % std::max<std::uint32_t>(1, options.episodes / 10) == 0) {
                    std::cout << "episode=" << episode << " reward=" << total_reward << " epsilon=" << epsilon
                              << " states=" << table.size() << " finished=" << result.finished << '\n';
                }
            }
            save(options.checkpoint, table, options);
            std::cout << "saved " << options.checkpoint << '\n';
        }
        evaluate(env, table, options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

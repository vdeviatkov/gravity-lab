#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace gravity_lab::classic {

constexpr std::size_t kObservationSize = 28;
constexpr std::int32_t kActionCount = 9;
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

struct Config {
    std::uint32_t level_group{0};
    std::uint32_t track{0};
    std::uint32_t league{0};
    std::uint32_t frame_skip{2};
    std::uint32_t max_episode_steps{5'000};
    std::uint64_t seed{1};
};

struct StepResult {
    Observation observation{};
    double reward{};
    bool terminated{};
    bool truncated{};
    bool finished{};
    bool crashed{};
    bool wheelie_finish{};
    std::int32_t physics_code{};
};

class Environment {
public:
    // An empty level pack path uses the levels.mrg embedded in the classic engine.
    explicit Environment(Config config = {}, std::filesystem::path level_pack = {});
    ~Environment();

    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    Environment(Environment&&) = delete;
    Environment& operator=(Environment&&) = delete;

    Observation reset();
    Observation reset(std::uint64_t seed);
    StepResult step(Action action);

    [[nodiscard]] Observation observation() const;
    [[nodiscard]] bool done() const noexcept;
    [[nodiscard]] bool terminated() const noexcept;
    [[nodiscard]] bool truncated() const noexcept;
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] bool crashed() const noexcept;
    [[nodiscard]] std::uint32_t episode_step() const noexcept;
    [[nodiscard]] const Config& config() const noexcept;
    [[nodiscard]] std::string track_name() const;
    [[nodiscard]] std::uint32_t track_count(std::uint32_t level_group) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gravity_lab::classic

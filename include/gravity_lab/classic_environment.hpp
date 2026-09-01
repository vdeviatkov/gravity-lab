#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace gravity_lab::classic {

class Renderer;

// Indices [0, kBaseObservationSize) are the original bike/track state (progress, league,
// per-component position/velocity relative to the bike's center body). Indices
// [kBaseObservationSize, kObservationSize) are an obstacle-distance sensor: kObstacleRayCount
// rays are cast from the bike's center, evenly spaced by full turns around it, and each entry is
// the distance (in [0, 1], where 1.0 means "no track segment within range") to the nearest
// bounded track polyline segment that ray intersects. See Environment::Impl::make_observation.
constexpr std::size_t kBaseObservationSize = 28;
constexpr std::size_t kObstacleRayCount = 8;
constexpr std::size_t kObservationSize = kBaseObservationSize + kObstacleRayCount;
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
    friend class Renderer;
    [[nodiscard]] void* native_physics_handle() noexcept;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gravity_lab::classic

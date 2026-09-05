#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace gravity_lab::classic {

class Renderer;

// The observation is laid out in fixed regions so that a model trained against a smaller region
// set is always reading an exact, unchanged prefix of what a model trained against more regions
// reads -- adding a region (or growing one) never changes the meaning of any existing index.
//
//   [0, kBaseObservationSize)                                  bike/track state: progress,
//       league, per-component position/velocity relative to the bike's center body.
//   [kBaseObservationSize, kBaseObservationSize + Config::obstacle_ray_count)
//                                                               obstacle-distance sensor: that
//       many rays cast from the bike's center, evenly spaced by full turns around it (ray count
//       changes the angle of every ray, but ray 0 always points along the direction of
//       increasing progress); each entry is the distance (in [0, 1], 1.0 = "nothing in range")
//       to the nearest bounded track polyline segment that ray intersects. Ray count is
//       per-environment (Config::obstacle_ray_count), not a fixed constant, so different trained
//       policies each keep their own ray count and angles exactly. Indices in
//       [kBaseObservationSize + Config::obstacle_ray_count, kObstacleRegionEnd) are left at zero
//       for a given environment.
//   [kObstacleRegionEnd, kAccelerationRegionEnd)                per-component acceleration: the
//       same 6 physics points as the base region, x/y acceleration each, always computed
//       regardless of obstacle_ray_count.
//   [kAccelerationRegionEnd, kTrackIdRegionEnd)                  track identity: a one-hot vector
//       over (level_group, track), index = level_group * kTracksPerLevelGroup + track, always
//       computed regardless of curriculum. Lets a single network condition its Q-values on which
//       of the 30 tracks it is on instead of inferring track identity purely from geometry.
//   [kTrackIdRegionEnd, kTrackIdRegionEnd + Config::obstacle_ray_count)
//                                                               head-clearance sensor: the same
//       ray count and angles as the obstacle-distance sensor above, but cast from physics point 5
//       (the rider's head -- the only point whose ground contact is an instant crash rather than a
//       graduated bounce, and the smallest of the three collision-radius classes; see
//       GamePhysics::const175_1_half and the docs/training-runs.md "SAC + REDQ" writeup for how
//       this was identified) instead of the center point. Each entry is
//       `max(0, distance - head_radius) / kObstacleMaxRange`, i.e. remaining clearance before the
//       head's own collision circle would touch the nearest track segment, not raw geometric
//       distance -- deliberately reuses Config::obstacle_ray_count rather than adding a second,
//       independently configurable ray count: the two sensors always share ray count and angles,
//       which keeps every existing derivation of ray count from a policy's declared
//       observation_size (see apps/ai_arcade.cpp, classic_policy_viewer.cpp) correct unmodified.
//       Indices in [kTrackIdRegionEnd + Config::obstacle_ray_count, kObservationSize) are left at
//       zero for a given environment, mirroring the obstacle-sensor region above.
//
// See Environment::Impl::make_observation.
constexpr std::size_t kBaseObservationSize = 28;
constexpr std::size_t kDefaultObstacleRayCount = 8;
constexpr std::size_t kMaxObstacleRayCount = 32;
constexpr std::size_t kObstacleRegionEnd = kBaseObservationSize + kMaxObstacleRayCount;
constexpr std::size_t kPhysicsPointCount = 6;
constexpr std::size_t kAccelerationSize = kPhysicsPointCount * 2;
constexpr std::size_t kAccelerationRegionEnd = kObstacleRegionEnd + kAccelerationSize;
constexpr std::size_t kLevelGroupCount = 3;
constexpr std::size_t kTracksPerLevelGroup = 10;
constexpr std::size_t kTrackIdSize = kLevelGroupCount * kTracksPerLevelGroup;
constexpr std::size_t kTrackIdRegionEnd = kAccelerationRegionEnd + kTrackIdSize;
// Reuses kMaxObstacleRayCount (the head-clearance sensor always uses the same ray count as the
// obstacle sensor -- see the layout comment above), kept as its own named constant for clarity at
// call sites that specifically mean the head-clearance region's reserved width.
constexpr std::size_t kMaxHeadClearanceRayCount = kMaxObstacleRayCount;
constexpr std::size_t kHeadClearanceRegionEnd = kTrackIdRegionEnd + kMaxHeadClearanceRayCount;
constexpr std::size_t kObservationSize = kHeadClearanceRegionEnd;
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
    // Number of obstacle-sensor rays this environment computes; must be in [1, kMaxObstacleRayCount].
    std::uint32_t obstacle_ray_count{static_cast<std::uint32_t>(kDefaultObstacleRayCount)};
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

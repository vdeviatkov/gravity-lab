#include "gravity_lab/classic_environment.hpp"

#include "GameLevel.h"
#include "GamePhysics.h"
#include "LevelLoader.h"
#include "TimerOrMotoPartOrMenuElem.h"
#include "class_10.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace gravity_lab::classic {
namespace {

constexpr double kFixed = 65'536.0;
// Same divisor used for the position-delta features below (indices 4..27), so obstacle-ray
// distances land in a comparable numeric range to the rest of the observation.
constexpr double kObstaclePositionScale = kFixed * 10.0;
constexpr double kObstacleMaxRange = kObstaclePositionScale * 5.0;
constexpr int kObstacleSearchRadius = 64;  // track segments scanned on each side of the bike
constexpr double kTwoPi = 6.283185307179586;
std::atomic<bool> active_environment{false};

struct Vec2 { double x; double y; };

double cross(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }

// GameLevel stores track polyline points scaled by 8 relative to the physics engine's F16
// (65536-scale) fixed-point coordinates; LevelLoader::method_90/91/92 apply the same `<< 1`
// correction when comparing level positions against a bike component's xF16/yF16.
Vec2 track_point(const GameLevel& level, int index) {
    return {static_cast<double>(level.pointPositions[index][0]) * 2.0,
            static_cast<double>(level.pointPositions[index][1]) * 2.0};
}

// Index of the track segment whose start point is at or immediately before world x `value`,
// via binary search (pointPositions is monotonically non-decreasing in x).
int locate_segment(const GameLevel& level, double value) {
    int low = 0;
    int high = level.pointsCount - 2;
    if (high <= low) return 0;
    while (low < high) {
        const int mid = low + (high - low + 1) / 2;
        if (track_point(level, mid).x <= value) low = mid; else high = mid - 1;
    }
    return low;
}

// Distance from `origin` along unit `direction` to the nearest intersection with a bounded
// track segment in [begin, end); segments are treated as finite (clipped to their own two
// endpoints), never as infinite lines. Returns kObstacleMaxRange if nothing is hit in range.
double cast_obstacle_ray(const Vec2& origin, const Vec2& direction, const GameLevel& level,
                         int begin, int end) {
    double nearest = kObstacleMaxRange;
    for (int i = begin; i < end; ++i) {
        const Vec2 start = track_point(level, i);
        const Vec2 segment{track_point(level, i + 1).x - start.x, track_point(level, i + 1).y - start.y};
        const double denominator = cross(direction, segment);
        if (std::abs(denominator) < 1e-9) continue;  // ray parallel to this segment
        const Vec2 originToStart{start.x - origin.x, start.y - origin.y};
        const double t = cross(originToStart, segment) / denominator;
        const double s = cross(originToStart, direction) / denominator;
        if (t >= 0.0 && t <= nearest && s >= 0.0 && s <= 1.0) nearest = t;
    }
    return nearest;
}

struct Controls { int drive; int lean; };

Controls decode(Action action) {
    switch (action) {
        case Action::Coast: return {0, 0};
        case Action::Throttle: return {1, 0};
        case Action::Brake: return {-1, 0};
        case Action::LeanBack: return {0, -1};
        case Action::LeanForward: return {0, 1};
        case Action::ThrottleLeanBack: return {1, -1};
        case Action::ThrottleLeanForward: return {1, 1};
        case Action::BrakeLeanBack: return {-1, -1};
        case Action::BrakeLeanForward: return {-1, 1};
    }
    throw std::invalid_argument("classic action is outside [0, 8]");
}

}  // namespace

struct Environment::Impl {
    Impl(Config selected_config, std::filesystem::path selected_pack)
        : config(std::move(selected_config)), level_pack(std::move(selected_pack)),
          loader(level_pack), physics(&loader) {
        if (config.level_group >= 3) throw std::invalid_argument("level_group must be in [0, 2]");
        if (config.league >= 4) throw std::invalid_argument("league must be in [0, 3]");
        if (config.frame_skip == 0 || config.frame_skip > 100) {
            throw std::invalid_argument("frame_skip must be in [1, 100]");
        }
        if (config.max_episode_steps == 0) throw std::invalid_argument("max_episode_steps must be positive");
        if (config.track >= loader.levelNames[config.level_group].size()) {
            throw std::invalid_argument("track index is outside the selected level group");
        }
        if (config.obstacle_ray_count == 0 || config.obstacle_ray_count > kMaxObstacleRayCount) {
            throw std::invalid_argument("obstacle_ray_count must be in [1, kMaxObstacleRayCount]");
        }
        physics.setMode(1);
        reset(config.seed);
    }

    Observation make_observation() const {
        Observation result{};
        const auto* center = physics.field_29[0]->motoComponents[5].get();
        const double start = static_cast<double>(loader.method_92());
        const double finish = static_cast<double>(loader.method_91());
        const double span = std::max(kFixed, finish - start);
        const double progress = (static_cast<double>(center->xF16) - start) / span;
        result[0] = progress;
        result[1] = 1.0 - progress;
        result[2] = physics.isTrackStarted() ? 0.0 : 1.0;
        result[3] = static_cast<double>(config.league) / 3.0;
        for (std::size_t i = 0; i < 6; ++i) {
            const auto* component = physics.field_29[i]->motoComponents[5].get();
            const std::size_t offset = 4 + i * 4;
            result[offset] = (component->xF16 - center->xF16) / (kFixed * 10.0);
            result[offset + 1] = (component->yF16 - center->yF16) / (kFixed * 10.0);
            result[offset + 2] = component->field_382 / (kFixed * 20.0);
            result[offset + 3] = component->field_383 / (kFixed * 20.0);
        }

        const Vec2 origin{static_cast<double>(center->xF16), static_cast<double>(center->yF16)};
        const GameLevel& level = *loader.gameLevel;
        const int segment_count = level.pointsCount - 1;
        const int current_segment = locate_segment(level, origin.x);
        const int begin = std::max(0, current_segment - kObstacleSearchRadius);
        const int end = std::min(segment_count, current_segment + kObstacleSearchRadius + 1);
        for (std::size_t ray = 0; ray < config.obstacle_ray_count; ++ray) {
            const double angle = kTwoPi * static_cast<double>(ray) / static_cast<double>(config.obstacle_ray_count);
            const Vec2 direction{std::cos(angle), std::sin(angle)};
            const double distance = cast_obstacle_ray(origin, direction, level, begin, end);
            result[kBaseObservationSize + ray] = distance / kObstacleMaxRange;
        }

        // Acceleration region: always computed, independent of obstacle_ray_count, so it lands
        // at the same fixed offset (kObstacleRegionEnd) regardless of ray count.
        for (std::size_t i = 0; i < kPhysicsPointCount; ++i) {
            const auto* component = physics.field_29[i]->motoComponents[5].get();
            const std::size_t offset = kObstacleRegionEnd + i * 2;
            result[offset] = component->field_385 / (kFixed * 20.0);
            result[offset + 1] = component->field_386 / (kFixed * 20.0);
        }
        return result;
    }

    Observation reset(std::uint64_t new_seed) {
        config.seed = new_seed;
        loader.method_88(static_cast<int>(config.level_group), static_cast<int>(config.track));
        physics.setMotoLeague(static_cast<int>(config.league));
        physics.disableGenerateInputAI();
        physics.method_30(0, 0);
        physics.method_53();
        steps = 0;
        terminal = time_limit = reached_finish = did_crash = wheelie = false;
        last_physics_code = 4;
        return make_observation();
    }

    double center_x() const {
        return physics.field_29[0]->motoComponents[5]->xF16 / kFixed;
    }

    Config config;
    std::filesystem::path level_pack;
    LevelLoader loader;
    GamePhysics physics;
    std::uint32_t steps{};
    bool terminal{};
    bool time_limit{};
    bool reached_finish{};
    bool did_crash{};
    bool wheelie{};
    int last_physics_code{4};
};

Environment::Environment(Config config, std::filesystem::path level_pack) {
    bool expected = false;
    if (!active_environment.compare_exchange_strong(expected, true)) {
        throw std::runtime_error("classic-v1 currently supports one active environment per process");
    }
    try {
        impl_ = std::make_unique<Impl>(config, std::move(level_pack));
    } catch (...) {
        active_environment = false;
        throw;
    }
}

Environment::~Environment() {
    impl_.reset();
    active_environment = false;
}

Observation Environment::reset() { return impl_->reset(impl_->config.seed); }
Observation Environment::reset(std::uint64_t seed) { return impl_->reset(seed); }

StepResult Environment::step(Action action) {
    if (done()) throw std::logic_error("classic step called after episode ended; call reset first");
    const Controls controls = decode(action);
    const double previous_x = impl_->center_x();
    impl_->physics.method_30(controls.drive, controls.lean);
    for (std::uint32_t i = 0; i < impl_->config.frame_skip; ++i) {
        impl_->last_physics_code = impl_->physics.updatePhysics();
        impl_->physics.method_53();
        if (impl_->last_physics_code == 1 || impl_->last_physics_code == 2) {
            impl_->terminal = impl_->reached_finish = true;
            impl_->wheelie = !impl_->physics.field_69;
            break;
        }
        if (impl_->last_physics_code == 3 || impl_->last_physics_code == 5) {
            impl_->terminal = impl_->did_crash = true;
            break;
        }
    }
    ++impl_->steps;
    if (!impl_->terminal && impl_->steps >= impl_->config.max_episode_steps) impl_->time_limit = true;

    double reward = (impl_->center_x() - previous_x) * 0.1 - 0.001;
    if (impl_->reached_finish) reward += 10.0;
    if (impl_->did_crash) reward -= 5.0;
    return {observation(), reward, impl_->terminal, impl_->time_limit, impl_->reached_finish,
            impl_->did_crash, impl_->wheelie, impl_->last_physics_code};
}

Observation Environment::observation() const { return impl_->make_observation(); }
bool Environment::done() const noexcept { return impl_->terminal || impl_->time_limit; }
bool Environment::terminated() const noexcept { return impl_->terminal; }
bool Environment::truncated() const noexcept { return impl_->time_limit; }
bool Environment::finished() const noexcept { return impl_->reached_finish; }
bool Environment::crashed() const noexcept { return impl_->did_crash; }
std::uint32_t Environment::episode_step() const noexcept { return impl_->steps; }
const Config& Environment::config() const noexcept { return impl_->config; }
std::string Environment::track_name() const {
    return impl_->loader.getName(static_cast<int>(impl_->config.level_group), static_cast<int>(impl_->config.track));
}
std::uint32_t Environment::track_count(std::uint32_t level_group) const {
    if (level_group >= impl_->loader.levelNames.size()) throw std::out_of_range("level_group must be in [0, 2]");
    return static_cast<std::uint32_t>(impl_->loader.levelNames[level_group].size());
}
void* Environment::native_physics_handle() noexcept { return &impl_->physics; }

}  // namespace gravity_lab::classic

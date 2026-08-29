#pragma once

#include "gravity_lab/map.hpp"
#include "gravity_lab/types.hpp"

#include <cstdint>

namespace gravity_lab {

class Environment {
public:
    Environment(Map map, Config config = {});

    Observation reset();
    Observation reset(std::uint64_t seed);
    StepResult step(Action action);

    [[nodiscard]] Observation observation() const;
    [[nodiscard]] const State& state() const noexcept { return state_; }
    [[nodiscard]] const Map& map() const noexcept { return map_; }
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] bool done() const noexcept { return terminated_ || truncated_; }
    [[nodiscard]] bool terminated() const noexcept { return terminated_; }
    [[nodiscard]] bool truncated() const noexcept { return truncated_; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }
    [[nodiscard]] bool crashed() const noexcept { return crashed_; }
    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

private:
    void integrate(Action action);
    [[nodiscard]] StepResult result(double reward) const;

    Map map_;
    Config config_;
    State state_{};
    std::uint64_t seed_{};
    bool terminated_{};
    bool truncated_{};
    bool finished_{};
    bool crashed_{};
};

}  // namespace gravity_lab

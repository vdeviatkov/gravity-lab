#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace gravity_lab {

struct TerrainPoint {
    double x{};
    double y{};
};

class Map {
public:
    static Map load(const std::filesystem::path& path);

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] double start_x() const noexcept { return start_x_; }
    [[nodiscard]] double finish_x() const noexcept { return finish_x_; }
    [[nodiscard]] double height_at(double x) const noexcept;
    [[nodiscard]] double slope_at(double x) const noexcept;
    [[nodiscard]] const std::vector<TerrainPoint>& points() const noexcept { return points_; }

private:
    std::string name_;
    double start_x_{};
    double finish_x_{};
    std::vector<TerrainPoint> points_;
};

}  // namespace gravity_lab

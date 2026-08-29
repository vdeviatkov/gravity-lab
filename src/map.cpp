#include "gravity_lab/map.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gravity_lab {
namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

double parse_number(const std::string& value, const std::string& field) {
    std::size_t consumed = 0;
    const double parsed = std::stod(trim(value), &consumed);
    if (consumed != trim(value).size() || !std::isfinite(parsed)) {
        throw std::runtime_error("invalid number for " + field + ": " + value);
    }
    return parsed;
}

}  // namespace

Map Map::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open map: " + path.string());

    Map map;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("map line " + std::to_string(line_number) + " is missing '='");
        }
        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        if (key == "name") {
            map.name_ = value;
        } else if (key == "start") {
            map.start_x_ = parse_number(value, key);
        } else if (key == "finish") {
            map.finish_x_ = parse_number(value, key);
        } else if (key == "point") {
            const auto comma = value.find(',');
            if (comma == std::string::npos) {
                throw std::runtime_error("map point on line " + std::to_string(line_number) + " must be x,y");
            }
            map.points_.push_back({parse_number(value.substr(0, comma), "point.x"),
                                   parse_number(value.substr(comma + 1), "point.y")});
        } else {
            throw std::runtime_error("unknown map field on line " + std::to_string(line_number) + ": " + key);
        }
    }

    if (map.name_.empty()) throw std::runtime_error("map requires a name");
    if (map.points_.size() < 2) throw std::runtime_error("map requires at least two points");
    if (!std::is_sorted(map.points_.begin(), map.points_.end(),
                        [](const auto& a, const auto& b) { return a.x < b.x; })) {
        throw std::runtime_error("map points must have strictly increasing x coordinates");
    }
    for (std::size_t i = 1; i < map.points_.size(); ++i) {
        if (map.points_[i - 1].x == map.points_[i].x) {
            throw std::runtime_error("map points must have unique x coordinates");
        }
    }
    if (map.start_x_ < map.points_.front().x || map.start_x_ >= map.finish_x_ ||
        map.finish_x_ > map.points_.back().x) {
        throw std::runtime_error("start and finish must lie within terrain, with start < finish");
    }
    return map;
}

double Map::height_at(double x) const noexcept {
    if (x <= points_.front().x) return points_.front().y;
    if (x >= points_.back().x) return points_.back().y;
    const auto upper = std::upper_bound(points_.begin(), points_.end(), x,
        [](double value, const TerrainPoint& point) { return value < point.x; });
    const auto lower = upper - 1;
    const double t = (x - lower->x) / (upper->x - lower->x);
    return lower->y + t * (upper->y - lower->y);
}

double Map::slope_at(double x) const noexcept {
    if (x <= points_.front().x) x = points_.front().x;
    if (x >= points_.back().x) x = std::nextafter(points_.back().x, points_.front().x);
    const auto upper = std::upper_bound(points_.begin(), points_.end(), x,
        [](double value, const TerrainPoint& point) { return value < point.x; });
    const auto lower = upper - 1;
    return std::atan2(upper->y - lower->y, upper->x - lower->x);
}

}  // namespace gravity_lab

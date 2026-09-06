#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace gravity_lab::classic {

class Environment;

// Optional SDL renderer for an existing faithful classic environment. Training,
// evaluation, and environment tests do not construct this class.
class Renderer {
public:
    explicit Renderer(Environment& environment, std::string title = "Gravity Lab Policy Viewer");
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    // Returns false after the window close button or Escape is pressed.
    bool render_frame(std::uint64_t elapsed_milliseconds = 0);
    void set_bike_only(bool enabled);
    // Render the entire level to PNG and write world-to-image bounds to JSON.
    void save_map_plate(const std::string& png_path, const std::string& json_path,
                        Environment& environment);
    void show_message(std::string message, std::uint32_t duration_milliseconds);
    [[nodiscard]] bool open() const noexcept;

    // Saves the currently-presented frame as a PNG. Intended for offscreen capture (e.g. under
    // SDL_VIDEODRIVER=dummy) to build a video after the fact; call once per render_frame(). Returns
    // false (rather than throwing) on a write failure, so a capture run can skip a bad frame and
    // continue rather than aborting a whole episode.
    bool save_frame(const std::string& path) const;

    // Legacy name: returns the last rendered viewport origin, not the bike center.
    // A world point (x,y) appears on screen at (x-origin.x, -y+origin.y).
    // Read-only: does not advance camera smoothing. With look-ahead disabled the
    // tracked bike reference is at screen (320,240) in the 640x480 viewport.
    [[nodiscard]] std::pair<int, int> bike_position() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gravity_lab::classic

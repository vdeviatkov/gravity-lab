#pragma once

#include <cstdint>
#include <memory>
#include <string>

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
    void show_message(std::string message, std::uint32_t duration_milliseconds);
    [[nodiscard]] bool open() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gravity_lab::classic

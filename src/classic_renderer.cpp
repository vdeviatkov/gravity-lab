#include "gravity_lab/classic_renderer.hpp"

#include "gravity_lab/classic_environment.hpp"

#include "GameCanvas.h"
#include "GamePhysics.h"
#include "Micro.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gravity_lab::classic {
namespace {
std::atomic<bool> active_renderer{false};
}

struct Renderer::Impl {
    Impl(Environment& environment, std::string title)
        : physics(static_cast<GamePhysics*>(environment.native_physics_handle())),
          previous_running(Micro::field_249), previous_menu(Micro::isInGameMenu) {
        Micro::field_249 = false;
        Micro::isInGameMenu = false;
        try {
            canvas = std::make_unique<GameCanvas>(&micro);
            micro.gameCanvas = canvas.get();
            micro.gamePhysics = physics;
            micro.menuManager = nullptr;
            micro.levelLoader = nullptr;
            canvas->init(physics);
            const int sprite_flags = canvas->loadSprites(3);
            physics->method_22(sprite_flags);
            canvas->requestRepaint(0);
            canvas->setViewPosition(-50, 150);
            canvas->setWindowTitle(title);
            Micro::field_249 = true;
            canvas->repaint();
        } catch (...) {
            canvas.reset();
            Micro::field_249 = previous_running;
            Micro::isInGameMenu = previous_menu;
            throw;
        }
    }

    ~Impl() {
        Micro::field_249 = false;
        canvas.reset();
        Micro::field_249 = previous_running;
        Micro::isInGameMenu = previous_menu;
    }

    GamePhysics* physics;
    bool previous_running;
    bool previous_menu;
    Micro micro;
    std::unique_ptr<GameCanvas> canvas;
};

Renderer::Renderer(Environment& environment, std::string title) {
    bool expected = false;
    if (!active_renderer.compare_exchange_strong(expected, true)) {
        throw std::runtime_error("classic renderer currently supports one active window per process");
    }
    try {
        impl_ = std::make_unique<Impl>(environment, std::move(title));
    } catch (...) {
        active_renderer = false;
        throw;
    }
}

Renderer::~Renderer() {
    impl_.reset();
    active_renderer = false;
}

bool Renderer::render_frame(std::uint64_t elapsed_milliseconds) {
    const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    impl_->micro.gameTimeMs = static_cast<std::int64_t>(std::min(elapsed_milliseconds, maximum));
    impl_->canvas->repaint();
    return impl_->canvas->isOpen();
}

void Renderer::show_message(std::string message, std::uint32_t duration_milliseconds) {
    const auto maximum = static_cast<std::uint32_t>(std::numeric_limits<int>::max());
    impl_->canvas->scheduleGameTimerTask(
        std::move(message), static_cast<int>(std::min(duration_milliseconds, maximum)));
}

bool Renderer::open() const noexcept { return impl_->canvas->isOpen(); }

}  // namespace gravity_lab::classic

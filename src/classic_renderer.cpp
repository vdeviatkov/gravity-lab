#include "gravity_lab/classic_renderer.hpp"

#include "gravity_lab/classic_environment.hpp"

#include "GameCanvas.h"
#include "GamePhysics.h"
#include "Micro.h"
#include "lcdui/CanvasImpl.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <fstream>
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

std::pair<int, int> Renderer::bike_position() const noexcept {
    // Read stored viewport offsets without advancing camera smoothing.
    return {-impl_->canvas->getDx(), impl_->canvas->addDy(0)};
}

bool Renderer::save_frame(const std::string& path) const {
    SDL_Renderer* renderer = impl_->canvas->getCanvasImpl()->getRenderer();
    const int width = impl_->canvas->getWidth();
    const int height = impl_->canvas->getHeight();
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_RGBA32);
    if (!surface) return false;
    const bool read_ok = SDL_RenderReadPixels(renderer, nullptr, SDL_PIXELFORMAT_RGBA32,
                                              surface->pixels, surface->pitch) == 0;
    const bool saved = read_ok && IMG_SavePNG(surface, path.c_str()) == 0;
    SDL_FreeSurface(surface);
    return saved;
}

}  // namespace gravity_lab::classic

namespace gravity_lab::classic {
void Renderer::set_bike_only(bool enabled) { impl_->canvas->bikeOnly = enabled; }

void Renderer::save_map_plate(const std::string& png_path, const std::string& json_path,
                              Environment& environment) {
    const auto points = environment.track_polyline();
    if (points.empty()) throw std::runtime_error("empty map geometry");
    int left = points.front().first, right = left;
    int bottom = points.front().second, top = bottom;
    for (const auto& [x, y] : points) {
        left = std::min(left, x); right = std::max(right, x);
        bottom = std::min(bottom, y); top = std::max(top, y);
    }
    left -= 100; right += 100; bottom -= 100; top += 160;
    const int width = (right - left + 1) / 2 * 2;
    const int height = (top - bottom + 1) / 2 * 2;
    SDL_Renderer* renderer = impl_->canvas->getCanvasImpl()->getRenderer();
    SDL_Texture* target = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                            SDL_TEXTUREACCESS_TARGET, width, height);
    if (!target) throw std::runtime_error(SDL_GetError());
    SDL_Texture* previous = SDL_GetRenderTarget(renderer);
    if (SDL_SetRenderTarget(renderer, target) != 0) {
        SDL_DestroyTexture(target); throw std::runtime_error(SDL_GetError());
    }
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    Graphics graphics(renderer);
    impl_->canvas->drawMap(&graphics, left, top, width, height, top);
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_RGBA32);
    const bool saved = surface && SDL_RenderReadPixels(renderer, nullptr, SDL_PIXELFORMAT_RGBA32,
        surface->pixels, surface->pitch) == 0 && IMG_SavePNG(surface, png_path.c_str()) == 0;
    SDL_FreeSurface(surface);
    SDL_SetRenderTarget(renderer, previous);
    SDL_DestroyTexture(target);
    if (!saved) throw std::runtime_error("failed to save map plate: " + png_path);
    std::ofstream metadata(json_path);
    metadata << "{\n  \"format\": \"gravity-lab-map-plate-v2\",\n"
             << "  \"min_ox\": " << left << ", \"min_oy\": " << -top
             << ",\n  \"width\": " << width << ", \"height\": " << height << "\n}\n";
    if (!metadata) throw std::runtime_error("failed to write plate metadata");
}
} // namespace gravity_lab::classic

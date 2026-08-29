#define SDL_MAIN_HANDLED
#include <SDL.h>

#include "gravity_lab/environment.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

constexpr int kWidth = 1100;
constexpr int kHeight = 650;
constexpr double kPixelsPerMeter = 45.0;

gravity_lab::Action action_from_keys(const Uint8* keys) {
    const bool throttle = keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_W];
    const bool brake = keys[SDL_SCANCODE_DOWN] || keys[SDL_SCANCODE_S];
    const bool back = keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A];
    const bool forward = keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D];
    if (throttle && back) return gravity_lab::Action::ThrottleLeanBack;
    if (throttle && forward) return gravity_lab::Action::ThrottleLeanForward;
    if (brake && back) return gravity_lab::Action::BrakeLeanBack;
    if (brake && forward) return gravity_lab::Action::BrakeLeanForward;
    if (throttle) return gravity_lab::Action::Throttle;
    if (brake) return gravity_lab::Action::Brake;
    if (back) return gravity_lab::Action::LeanBack;
    if (forward) return gravity_lab::Action::LeanForward;
    return gravity_lab::Action::Coast;
}

void line(SDL_Renderer* renderer, double x1, double y1, double x2, double y2,
          double camera_x, double camera_y) {
    const auto sx = [camera_x](double x) { return static_cast<int>(kWidth / 2.0 + (x - camera_x) * kPixelsPerMeter); };
    const auto sy = [camera_y](double y) { return static_cast<int>(kHeight * 0.68 - (y - camera_y) * kPixelsPerMeter); };
    SDL_RenderDrawLine(renderer, sx(x1), sy(y1), sx(x2), sy(y2));
}

void render(SDL_Renderer* renderer, const gravity_lab::Environment& env) {
    const auto& state = env.state();
    const double camera_x = std::max(env.map().start_x() + 8.0, state.x + 4.0);
    const double camera_y = env.map().height_at(state.x);
    SDL_SetRenderDrawColor(renderer, 18, 24, 38, 255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 100, 215, 130, 255);
    const auto& points = env.map().points();
    for (std::size_t i = 1; i < points.size(); ++i) {
        line(renderer, points[i - 1].x, points[i - 1].y, points[i].x, points[i].y, camera_x, camera_y);
    }

    SDL_SetRenderDrawColor(renderer, 255, 210, 80, 255);
    line(renderer, env.map().finish_x(), env.map().height_at(env.map().finish_x()),
         env.map().finish_x(), env.map().height_at(env.map().finish_x()) + 3.0, camera_x, camera_y);

    const double c = std::cos(state.angle);
    const double s = std::sin(state.angle);
    const double rear_x = state.x - 0.62 * c;
    const double rear_y = state.y - 0.62 * s - 0.18;
    const double front_x = state.x + 0.62 * c;
    const double front_y = state.y + 0.62 * s - 0.18;
    SDL_SetRenderDrawColor(renderer, 235, 238, 245, 255);
    line(renderer, rear_x, rear_y, front_x, front_y, camera_x, camera_y);
    line(renderer, rear_x, rear_y, state.x, state.y + 0.28, camera_x, camera_y);
    line(renderer, state.x, state.y + 0.28, front_x, front_y, camera_x, camera_y);
    SDL_SetRenderDrawColor(renderer, 255, 100, 90, 255);
    line(renderer, state.x, state.y + 0.28, state.x - 0.15 * s, state.y + 0.72 * c, camera_x, camera_y);

    SDL_RenderPresent(renderer);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path map = argc > 1 ? argv[1] : "maps/hills.gdmap";
        gravity_lab::Environment env(gravity_lab::Map::load(map));
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) throw std::runtime_error(SDL_GetError());
        SDL_Window* window = SDL_CreateWindow("Gravity Lab", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                               kWidth, kHeight, SDL_WINDOW_SHOWN);
        if (!window) throw std::runtime_error(SDL_GetError());
        SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
        if (!renderer) throw std::runtime_error(SDL_GetError());

        bool running = true;
        Uint64 previous = SDL_GetPerformanceCounter();
        double accumulator = 0.0;
        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) running = false;
                if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) running = false;
                if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_r) env.reset();
            }
            const Uint64 now = SDL_GetPerformanceCounter();
            accumulator += static_cast<double>(now - previous) / SDL_GetPerformanceFrequency();
            previous = now;
            const Uint8* keys = SDL_GetKeyboardState(nullptr);
            while (accumulator >= 1.0 / 30.0) {
                if (!env.done()) env.step(action_from_keys(keys));
                accumulator -= 1.0 / 30.0;
            }
            render(renderer, env);
        }
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        SDL_Quit();
        return 1;
    }
}

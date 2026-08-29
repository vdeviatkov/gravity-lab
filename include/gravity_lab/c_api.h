#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(gravity_lab_c_EXPORTS)
#    define GD_API __declspec(dllexport)
#  else
#    define GD_API __declspec(dllimport)
#  endif
#else
#  define GD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { GD_OBSERVATION_SIZE = 12, GD_ACTION_COUNT = 9 };

typedef enum gd_action {
    GD_COAST = 0,
    GD_THROTTLE = 1,
    GD_BRAKE = 2,
    GD_LEAN_BACK = 3,
    GD_LEAN_FORWARD = 4,
    GD_THROTTLE_LEAN_BACK = 5,
    GD_THROTTLE_LEAN_FORWARD = 6,
    GD_BRAKE_LEAN_BACK = 7,
    GD_BRAKE_LEAN_FORWARD = 8
} gd_action;

typedef struct gd_env gd_env;

typedef struct gd_config {
    double time_step;
    uint32_t frame_skip;
    uint32_t max_episode_steps;
    uint64_t seed;
} gd_config;

typedef struct gd_step_result {
    double observation[GD_OBSERVATION_SIZE];
    double reward;
    int terminated;
    int truncated;
    int finished;
    int crashed;
} gd_step_result;

GD_API gd_config gd_default_config(void);
GD_API gd_env* gd_create(const char* map_path, gd_config config);
GD_API void gd_destroy(gd_env* env);
GD_API int gd_reset(gd_env* env, uint64_t seed, double observation[GD_OBSERVATION_SIZE]);
GD_API int gd_step(gd_env* env, int action, gd_step_result* result);
GD_API const char* gd_last_error(void);

#ifdef __cplusplus
}
#endif

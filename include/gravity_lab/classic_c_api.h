#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(gravity_lab_classic_EXPORTS)
#    define GDC_API __declspec(dllexport)
#  else
#    define GDC_API __declspec(dllimport)
#  endif
#else
#  define GDC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { GDC_OBSERVATION_SIZE = 134, GDC_ACTION_COUNT = 9 };

typedef struct gdc_env gdc_env;

typedef struct gdc_config {
    uint32_t level_group;
    uint32_t track;
    uint32_t league;
    uint32_t frame_skip;
    uint32_t max_episode_steps;
    uint64_t seed;
    uint32_t obstacle_ray_count;
} gdc_config;

typedef struct gdc_step_result {
    double observation[GDC_OBSERVATION_SIZE];
    int terminated;
    int truncated;
    int finished;
    int crashed;
    int wheelie_finish;
    int physics_code;
} gdc_step_result;

GDC_API gdc_config gdc_default_config(void);
GDC_API gdc_env* gdc_create(const char* level_pack_path, gdc_config config);
GDC_API void gdc_destroy(gdc_env* env);
GDC_API int gdc_reset(gdc_env* env, uint64_t seed, double observation[GDC_OBSERVATION_SIZE]);
GDC_API int gdc_step(gdc_env* env, int action, gdc_step_result* result);
GDC_API const char* gdc_track_name(gdc_env* env);
GDC_API int gdc_track_count(gdc_env* env, uint32_t level_group, uint32_t* count);
// The render camera's tracked bike position (fixed-point pixels). Not part of the observation or
// training contract -- a diagnostic accessor for visualization tooling (plotting a rollout's path).
GDC_API int gdc_bike_position(gdc_env* env, int* x, int* y);
// The current track's ground polyline: vertex count, then each vertex's (x, y) in the same
// fixed-point pixel space as gdc_bike_position. Not part of the observation/training contract -- a
// diagnostic accessor for visualization tooling (drawing the track once instead of stitching it
// together from rendered frames). Call gdc_track_polyline_count first to size the output buffers.
GDC_API int gdc_track_polyline_count(gdc_env* env, uint32_t* count);
GDC_API int gdc_track_polyline_points(gdc_env* env, int* xs, int* ys, uint32_t count);
// The current track's start/finish flag positions, same coordinate space as the above.
GDC_API int gdc_track_start_finish(gdc_env* env, int* start_x, int* start_y, int* finish_x, int* finish_y);
GDC_API const char* gdc_last_error(void);

#ifdef __cplusplus
}
#endif

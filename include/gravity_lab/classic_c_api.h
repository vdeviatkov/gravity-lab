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

enum { GDC_OBSERVATION_SIZE = 72, GDC_ACTION_COUNT = 9 };

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
    double reward;
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
GDC_API const char* gdc_last_error(void);

#ifdef __cplusplus
}
#endif

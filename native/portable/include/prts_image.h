#ifndef PRTS_IMAGE_H
#define PRTS_IMAGE_H
#include <stdint.h>
#if defined(_WIN32)
#define PRTS_IMAGE_API __declspec(dllexport)
#else
#define PRTS_IMAGE_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* RGB24 upright input -> float32 CHW letterbox, pad=114. mean/std may be NULL.
 * Output capacity = 3*target_width*target_height. crop = x,y,resized_w,resized_h.
 * Bilinear uint8 resampling can differ by one quantization unit from OpenCV;
 * compare model outputs before adopting a new preprocessing implementation.
 */
PRTS_IMAGE_API int32_t prts_letterbox_rgb(const uint8_t *rgb,int32_t width,int32_t height,
    int32_t target_width,int32_t target_height,const float *mean,const float *std,
    float *output,int32_t *crop);
/* Decode positive CHW class probabilities from the exported Mask2Former.
 * Crop in model-input pixels, then resize probabilities before class argmax.
 * All output grids have grid_width*grid_height elements. walkable is the final
 * class-membership AND summed-probability>=0.5 mask used by the planner.
 */
PRTS_IMAGE_API int32_t prts_decode_semantic(const float *probabilities,int32_t classes,
    int32_t score_width,int32_t score_height,const int32_t *crop,int32_t input_width,int32_t input_height,
    int32_t grid_width,int32_t grid_height,const int32_t *walk_ids,int32_t walk_count,
    uint8_t *class_ids,float *walk_probability,float *confidence,uint8_t *walkable);
#ifdef __cplusplus
}
#endif
#endif

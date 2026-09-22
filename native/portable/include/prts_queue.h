#ifndef PRTS_QUEUE_H
#define PRTS_QUEUE_H
#include <stdint.h>
#if defined(_WIN32)
#define PRTS_QUEUE_API __declspec(dllexport)
#else
#define PRTS_QUEUE_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Split digits-only OCR rows by visible gaps; no requested target or new digits.
 * quad = TL/TR/BR/BL upright RGB pixel coordinates.
 * ranges capacity = 2*digit_count [start,end) pairs; boxes = 8*digit_count floats.
 * Returns group count (1 preserves original) or -1 for invalid input.
 */
PRTS_QUEUE_API int32_t prts_queue_split_rgb(const uint8_t *rgb,int32_t width,int32_t height,
    const float *quad,int32_t digit_count,int32_t *ranges,float *boxes);
#ifdef __cplusplus
}
#endif
#endif

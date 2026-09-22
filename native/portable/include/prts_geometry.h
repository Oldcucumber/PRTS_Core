#ifndef PRTS_GEOMETRY_H
#define PRTS_GEOMETRY_H
#include <stdint.h>
#if defined(_WIN32) && defined(PRTS_GEOMETRY_BUILD)
#define PRTS_GEOMETRY_API __declspec(dllexport)
#else
#define PRTS_GEOMETRY_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* Row-major binary image mask. Coordinates are pixel indices, not metres.
 * anchor_x=-1 selects the closest valid near-field point to the image centre.
 * preference is -1 / 0 / 1 and is only a lateral image preference.
 * Prior path softly changes cost; it cannot make a blocked pixel traversable.
 * Caller owns all arrays. Return point count, 0 for no path, -1 for bad input,
 * -2 when output capacity is insufficient. Capacity width*height always suffices.
 */
PRTS_GEOMETRY_API int32_t prts_corridor_search(
    const uint8_t *free_mask, int32_t width, int32_t height,
    int32_t anchor_x, int32_t preference, const int32_t *prior_xy, int32_t prior_count,
    int32_t *output_xy, int32_t capacity, double *coverage);

/* Entire polyline, including pixels between vertices. Returns outside count;
 * -1 for invalid arguments. Empty paths have checked_pixels=0 and are not valid.
 */
PRTS_GEOMETRY_API int32_t prts_path_validate(
    const uint8_t *free_mask, int32_t width, int32_t height,
    const int32_t *xy, int32_t count, int32_t *checked_pixels);

/* Fixed normalized lookahead rows, matching the Python reference.
 * angles[0:3] = overall / near / far angle from image vertical in degrees.
 */
PRTS_GEOMETRY_API int32_t prts_path_heading(
    const int32_t *xy, int32_t count, int32_t width, int32_t height, double *angles);
#ifdef __cplusplus
}
#endif
#endif

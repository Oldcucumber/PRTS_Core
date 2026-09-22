#ifndef PRTS_VLM_H
#define PRTS_VLM_H
#include <stddef.h>
#include <stdint.h>
#include "prts_geometry.h"
#include "prts_onnx.h"
#include "prts_image.h"
#include "prts_queue.h"
#if defined(_WIN32)
#define PRTS_API __declspec(dllexport)
#else
#define PRTS_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct prts_vlm prts_vlm;
typedef struct {
    const char * model_path;
    const char * projector_path;
    int32_t context_tokens;
    int32_t threads;
    int32_t image_slices;
    int32_t use_gpu;
} prts_vlm_config;

/* Byte pieces can split a UTF-8 character; consumers use an incremental decoder.
 * Callback executes on the caller's inference thread and must not reenter run.
 * One run per handle. cancel may be called concurrently. destroy waits for run. */
typedef void (*prts_text_callback)(const uint8_t * bytes, size_t length, void * user);
PRTS_API prts_vlm * prts_vlm_create(const prts_vlm_config * config);
/* RGB24 contiguous image is optional (NULL); input is borrowed for this call.
 * Returns 0 on end of response, 1 when cancelled, 2 on output token limit,
 * or -1 on error. A fresh context per request bounds memory and stale history. */
PRTS_API int32_t prts_vlm_run(prts_vlm * handle, const char * prompt,
    const uint8_t * rgb, uint32_t width, uint32_t height, int32_t max_tokens,
    prts_text_callback callback, void * user);
/* Official-runtime build: same call with schema-constrained output. JSON schema
 * limits structure; it does not decide whether a waiting event occurred. */
PRTS_API int32_t prts_vlm_run_json(prts_vlm * handle, const char * prompt, const char * json_schema,
    const uint8_t * rgb, uint32_t width, uint32_t height, int32_t max_tokens,
    prts_text_callback callback, void * user);
PRTS_API void prts_vlm_cancel(prts_vlm * handle);
/* Configure between turns. Deterministic mode uses greedy decoding, no presence penalty. */
PRTS_API void prts_vlm_set_deterministic(prts_vlm * handle, int32_t enabled);
PRTS_API void prts_vlm_destroy(prts_vlm * handle);
/* Thread-local last error; valid until next operation on this thread. */
PRTS_API const char * prts_vlm_last_error(void);
PRTS_API const char * prts_vlm_runtime_revision(void);

#ifdef __cplusplus
}
#endif
#endif

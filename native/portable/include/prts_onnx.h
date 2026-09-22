#ifndef PRTS_ONNX_H
#define PRTS_ONNX_H
#include <stdint.h>
#if defined(_WIN32)
#define PRTS_ONNX_API __declspec(dllexport)
#else
#define PRTS_ONNX_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct prts_onnx prts_onnx;
typedef struct {
    const float *data;
    int64_t element_count;
    int32_t rank;
    int64_t shape[8];
} prts_float_tensor;

/* Inject OrtGetApiBase() from the app's SINGLE linked ONNX Runtime. This
 * bridge does not link or load a second runtime. API version 23 is requested.
 * Model has one float32 input and one float32 output. CPU is the reference;
 * use_coreml=1 explicitly registers CoreML and returns error if unavailable.
 */
PRTS_ONNX_API prts_onnx *prts_onnx_create(const void *ort_api_base,
    const char *model_path_utf8,int32_t threads,int32_t use_coreml);
/* Input is borrowed for Run. Output is owned by the handle and remains valid
 * until its next Run or destroy. Serialize Run and copy output before reuse.
 * Returns 0 on success or -1 with thread-local last_error.
 */
PRTS_ONNX_API int32_t prts_onnx_run(prts_onnx *handle,const float *input,
    int64_t element_count,const int64_t *shape,int32_t rank,prts_float_tensor *output);
PRTS_ONNX_API const char *prts_onnx_last_error(void);
PRTS_ONNX_API void prts_onnx_destroy(prts_onnx *handle);
#ifdef __cplusplus
}
#endif
#endif

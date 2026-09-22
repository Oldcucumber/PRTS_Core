#include "prts_onnx.h"
#include "onnxruntime_c_api.h"
#include <memory>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

static thread_local std::string last_error;
struct prts_onnx {
    const OrtApi *api=nullptr;
    OrtEnv *env=nullptr;
    OrtSessionOptions *options=nullptr;
    OrtSession *session=nullptr;
    OrtMemoryInfo *memory=nullptr;
    OrtValue *input=nullptr,*output=nullptr;
    OrtTensorTypeAndShapeInfo *output_info=nullptr;
    OrtAllocator *allocator=nullptr;
    char *input_name=nullptr,*output_name=nullptr;
    void check(OrtStatus *status) {
        if(!status)return;
        std::string message=api->GetErrorMessage(status);api->ReleaseStatus(status);
        throw std::runtime_error(message);
    }
    void clear_values() {
        if(output_info){api->ReleaseTensorTypeAndShapeInfo(output_info);output_info=nullptr;}
        if(input){api->ReleaseValue(input);input=nullptr;}
        if(output){api->ReleaseValue(output);output=nullptr;}
    }
    ~prts_onnx() {
        if(!api)return;clear_values();
        if(input_name)allocator->Free(allocator,input_name);
        if(output_name)allocator->Free(allocator,output_name);
        if(session)api->ReleaseSession(session);
        if(options)api->ReleaseSessionOptions(options);
        if(memory)api->ReleaseMemoryInfo(memory);
        if(env)api->ReleaseEnv(env);
    }
};

prts_onnx *prts_onnx_create(const void *base,const char *model,int32_t threads,int32_t coreml) {
    last_error.clear();
    try {
        if(!base || !model || threads<1)throw std::runtime_error("Invalid ONNX configuration");
        auto h=std::make_unique<prts_onnx>();
        h->api=static_cast<const OrtApiBase *>(base)->GetApi(ORT_API_VERSION);
        if(!h->api)throw std::runtime_error("ONNX Runtime does not support API version 23");
        h->check(h->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"PRTS",&h->env));
        h->check(h->api->CreateSessionOptions(&h->options));
        h->check(h->api->SetIntraOpNumThreads(h->options,threads));
        h->check(h->api->SetInterOpNumThreads(h->options,1));
        h->check(h->api->DisableCpuMemArena(h->options));
        h->check(h->api->DisableMemPattern(h->options));
        h->check(h->api->AddSessionConfigEntry(h->options,"session.disable_prepacking","1"));
        if(coreml) {
            const char *keys[]={"ModelFormat","MLComputeUnits","RequireStaticInputShapes","EnableOnSubgraphs"};
            const char *values[]={"MLProgram","ALL","0","0"};
            h->check(h->api->SessionOptionsAppendExecutionProvider(h->options,"CoreML",keys,values,4));
        }
#ifdef _WIN32
        int length=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,model,-1,nullptr,0);
        if(!length)throw std::runtime_error("Model path is not valid UTF-8");
        std::wstring wide(length,L'\0');MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,model,-1,wide.data(),length);
        h->check(h->api->CreateSession(h->env,wide.c_str(),h->options,&h->session));
#else
        h->check(h->api->CreateSession(h->env,model,h->options,&h->session));
#endif
        size_t inputs=0,outputs=0;
        h->check(h->api->SessionGetInputCount(h->session,&inputs));
        h->check(h->api->SessionGetOutputCount(h->session,&outputs));
        if(inputs!=1 || outputs!=1)throw std::runtime_error("This bridge expects one input and one output");
        h->check(h->api->GetAllocatorWithDefaultOptions(&h->allocator));
        h->check(h->api->SessionGetInputName(h->session,0,h->allocator,&h->input_name));
        h->check(h->api->SessionGetOutputName(h->session,0,h->allocator,&h->output_name));
        h->check(h->api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&h->memory));
        return h.release();
    } catch(const std::exception& error) { last_error=error.what();return nullptr; }
}

int32_t prts_onnx_run(prts_onnx *h,const float *data,int64_t elements,const int64_t *shape,
                     int32_t rank,prts_float_tensor *out) {
    last_error.clear();if(out)*out={};
    try {
        if(!h || !data || !shape || !out || rank<1 || rank>8 || elements<1)
            throw std::runtime_error("Invalid ONNX tensor");
        int64_t total=1;
        for(int i=0;i<rank;++i) {
            if(shape[i]<1 || total>elements/shape[i])throw std::runtime_error("Tensor shape exceeds input buffer");
            total*=shape[i];
        }
        if(total!=elements)throw std::runtime_error("Tensor shape does not match input buffer");
        h->clear_values();
        h->check(h->api->CreateTensorWithDataAsOrtValue(h->memory,const_cast<float*>(data),
            size_t(elements)*sizeof(float),shape,rank,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&h->input));
        const char *input_names[]={h->input_name},*output_names[]={h->output_name};
        const OrtValue *inputs[]={h->input};
        h->check(h->api->Run(h->session,nullptr,input_names,inputs,1,output_names,1,&h->output));
        h->check(h->api->GetTensorTypeAndShape(h->output,&h->output_info));
        ONNXTensorElementDataType type;size_t output_rank=0,output_elements=0;
        h->check(h->api->GetTensorElementType(h->output_info,&type));
        h->check(h->api->GetDimensionsCount(h->output_info,&output_rank));
        if(type!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || output_rank>8)
            throw std::runtime_error("Expected float32 output with rank <= 8");
        h->check(h->api->GetDimensions(h->output_info,out->shape,output_rank));
        h->check(h->api->GetTensorShapeElementCount(h->output_info,&output_elements));
        void *values=nullptr;h->check(h->api->GetTensorMutableData(h->output,&values));
        out->data=static_cast<float*>(values);out->rank=int32_t(output_rank);out->element_count=int64_t(output_elements);
        return 0;
    } catch(const std::exception& error) { last_error=error.what();return -1; }
}
const char *prts_onnx_last_error() { return last_error.c_str(); }
void prts_onnx_destroy(prts_onnx *handle) { delete handle; }

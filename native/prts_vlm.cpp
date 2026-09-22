#include "prts_vlm.h"
#include "llama.h"
#include "common.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#ifdef PRTS_OFFICIAL_RUNTIME
#include "sampling.h"
#include "json-schema-to-grammar.h"
#endif
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

static thread_local std::string last_error;
static std::once_flag backend_once;

struct prts_vlm {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    mtmd_context * vision = nullptr;
    common_chat_templates_ptr templates;
    std::atomic<bool> cancelled{false};
    bool deterministic = false;
    std::mutex run_mutex;
    ~prts_vlm() {
        if (vision) mtmd_free(vision);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

static bool abort_requested(void * user) {
    return static_cast<prts_vlm *>(user)->cancelled.load();
}

extern "C" {
PRTS_API prts_vlm * prts_vlm_create(const prts_vlm_config * config) {
    try {
        last_error.clear();
        if (!config || !config->model_path || !config->projector_path)
            throw std::runtime_error("Model and projector paths are required");
        std::call_once(backend_once, [] { llama_backend_init(); });
        auto handle = std::make_unique<prts_vlm>();
        auto mp = llama_model_default_params();
#if defined(PRTS_OFFICIAL_RUNTIME) && !defined(PRTS_CPU_REPACK_ENABLED)
        mp.use_extra_bufts = false;
#endif
        mp.n_gpu_layers = config->use_gpu ? 99 : 0;
        handle->model = llama_model_load_from_file(config->model_path, mp);
        if (!handle->model) throw std::runtime_error("Cannot load language GGUF");
        auto cp = llama_context_default_params();
        cp.n_ctx = config->context_tokens > 0 ? config->context_tokens : 4096;
        cp.n_batch = 512;
        cp.n_ubatch = PRTS_VLM_MICROBATCH;
        cp.n_threads = cp.n_threads_batch = config->threads > 0 ? config->threads : 4;
        cp.n_seq_max = 1;
        cp.abort_callback = abort_requested;
        cp.abort_callback_data = handle.get();
        handle->context = llama_init_from_model(handle->model, cp);
        if (!handle->context) throw std::runtime_error("Cannot create language context");
        auto vp = mtmd_context_params_default();
        vp.use_gpu = config->use_gpu != 0;
        vp.n_threads = cp.n_threads;
        vp.warmup = false;
#ifdef PRTS_OFFICIAL_RUNTIME
        vp.image_max_tokens = 512;
#else
        vp.image_max_slice_nums = config->image_slices > 0 ? config->image_slices : 4;
#endif
        handle->vision = mtmd_init_from_file(config->projector_path, handle->model, vp);
        if (!handle->vision) throw std::runtime_error("Cannot load vision projector");
        handle->templates = common_chat_templates_init(handle->model, "");
        return handle.release();
    } catch (const std::exception & error) {
        last_error = error.what();
        return nullptr;
    }
}

static int32_t run_impl(prts_vlm * handle, const char * prompt, const char * schema,
        const uint8_t * rgb, uint32_t width, uint32_t height, int32_t max_tokens,
        prts_text_callback callback, void * user) {
    if (!handle || !prompt) { last_error = "Handle and prompt are required"; return -1; }
    std::lock_guard<std::mutex> guard(handle->run_mutex);
    try {
        last_error.clear();
        handle->cancelled.store(false);
        llama_memory_clear(llama_get_memory(handle->context), true);
        common_chat_templates_inputs input;
        input.use_jinja = true;
        input.enable_thinking = false;
        input.chat_template_kwargs["enable_thinking"] = "false";
        common_chat_msg message;
        message.role = "user";
        message.content = (rgb ? std::string(mtmd_default_marker()) + "\n" : "") + prompt;
        input.messages.push_back(std::move(message));
        auto formatted = common_chat_templates_apply(handle->templates.get(), input);
#ifdef PRTS_OFFICIAL_RUNTIME
        mtmd_input_text text{formatted.prompt.c_str(), formatted.prompt.size(), true, true};
#else
        mtmd_input_text text{formatted.prompt.c_str(), true, true};
#endif
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            rgb ? mtmd_bitmap_init(width, height, rgb) : nullptr, mtmd_bitmap_free);
        const mtmd_bitmap * bitmap_ptr = bitmap.get();
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            mtmd_input_chunks_init(), mtmd_input_chunks_free);
        int result = mtmd_tokenize(handle->vision, chunks.get(), &text, rgb ? &bitmap_ptr : nullptr, rgb ? 1 : 0);
        if (result) throw std::runtime_error("Multimodal tokenization failed");
        if (mtmd_helper_get_n_pos(chunks.get()) + max_tokens >= llama_n_ctx(handle->context))
            throw std::runtime_error("Image and prompt exceed the configured context budget");
        llama_pos position = 0;
        result = mtmd_helper_eval_chunks(handle->vision, handle->context, chunks.get(), 0, 0, 512, true, &position);
        if (handle->cancelled.load()) return 1;
        if (result) throw std::runtime_error("Multimodal prompt evaluation failed");
        auto vocab = llama_model_get_vocab(handle->model);
#ifdef PRTS_OFFICIAL_RUNTIME
        common_params_sampling params;
        params.seed=42;params.temp=0.7f;params.top_p=0.8f;params.top_k=20;
        params.min_p=0.0f;params.penalty_present=1.5f;params.penalty_repeat=1.0f;
        if (handle->deterministic) { params.temp=0.0f;params.penalty_present=0.0f; }
        if (schema && *schema) {
            params.grammar=common_grammar(COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
                json_schema_to_grammar(common_json::parse(schema)));
        }
        common_sampler_ptr sampler(common_sampler_init(handle->model, params));
#else
        if (schema && *schema) throw std::runtime_error("Structured output needs the official runtime build");
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
#endif
        for (int i = 0; i < max_tokens; ++i) {
            if (handle->cancelled.load()) return 1;
#ifdef PRTS_OFFICIAL_RUNTIME
            llama_token token=common_sampler_sample(sampler.get(),handle->context,-1);
            common_sampler_accept(sampler.get(),token,true);
#else
            llama_token token = llama_sampler_sample(sampler.get(), handle->context, -1);
#endif
            if (llama_vocab_is_eog(vocab, token)) return 0;
            auto piece = common_token_to_piece(handle->context, token);
            if (callback) callback(reinterpret_cast<const uint8_t *>(piece.data()), piece.size(), user);
            auto batch = llama_batch_get_one(&token, 1);
            result = llama_decode(handle->context, batch);
            if (handle->cancelled.load()) return 1;
            if (result) throw std::runtime_error("Response token evaluation failed");
        }
        return 2;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

PRTS_API int32_t prts_vlm_run(prts_vlm * handle,const char * prompt,const uint8_t * rgb,
        uint32_t width,uint32_t height,int32_t max_tokens,prts_text_callback callback,void * user) {
    return run_impl(handle,prompt,nullptr,rgb,width,height,max_tokens,callback,user);
}
PRTS_API int32_t prts_vlm_run_json(prts_vlm * handle,const char * prompt,const char * schema,const uint8_t * rgb,
        uint32_t width,uint32_t height,int32_t max_tokens,prts_text_callback callback,void * user) {
    return run_impl(handle,prompt,schema,rgb,width,height,max_tokens,callback,user);
}

PRTS_API void prts_vlm_cancel(prts_vlm * handle) {
    if (handle) handle->cancelled.store(true);
}
PRTS_API void prts_vlm_set_deterministic(prts_vlm * handle, int32_t enabled) {
    if (!handle) return;
    std::lock_guard<std::mutex> guard(handle->run_mutex);
    handle->deterministic = enabled != 0;
}
PRTS_API void prts_vlm_destroy(prts_vlm * handle) {
    if (!handle) return;
    handle->cancelled.store(true);
    { std::lock_guard<std::mutex> guard(handle->run_mutex); }
    delete handle;
}
PRTS_API const char * prts_vlm_last_error(void) { return last_error.c_str(); }
PRTS_API const char * prts_vlm_runtime_revision(void) {
#ifdef PRTS_OFFICIAL_RUNTIME
    return "llama.cpp@b29c606e28a01b1bc8c1351026a0fa6e616bf6c4";
#else
    return "llama.cpp-omni@13401aa8480d68a725e131cc4dbe5eda1f996a7f";
#endif
}
}

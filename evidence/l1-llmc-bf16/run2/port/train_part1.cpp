/*
GPT-2 Transformer Neural Net training loop - SYCL 2020 port for Intel oneAPI DPC++.
Ported from llm.c's train_gpt2.cu (CUDA). Single device, bf16 mixed precision
(bf16 parameters/activations, fp32 master weights and accumulation), oneMath BLAS.
*/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <cmath>
// ----------- CPU utilities -----------
#include "llmc/utils.h"
#include "llmc/tokenizer.h"
#include "llmc/dataloader.h"
#include "llmc/rand.h"
#include "llmc/schedulers.h"
#include "llmc/sampler.h"
#include "llmc/logger.h"
#include "llmc/mfu.h"
#include "llmc/outlier_detector.h"
// ----------- SYCL utilities -----------
#include "llmc/sycl_common.h"
#include "llmc/sycl_utils.h"
#include "llmc/blas_common.h"
// ----------- Layer implementations in SYCL -----------
#include "llmc/encoder.h"
#include "llmc/layernorm.h"
#include "llmc/matmul.h"
#include "llmc/attention.h"
#include "llmc/fused_classifier.h"
#include "llmc/adamw.h"
#include "llmc/global_norm.h"
// ----------- Single-device (no NCCL/MPI) support -----------
#include "llmc/zero.h"

// ----------------------------------------------------------------------------
// global vars for I/O
char filename_buffer[512];

// ----------------------------------------------------------------------------
// global SYCL queue for this process (single device)
sycl::queue sycl_queue;
static std::string sycl_device_name = "unknown";

// buffer size to use for device <-> disk io
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
constexpr const int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte; // (V, C)
    floatX* wpe; // (maxT, C)
    floatX* ln1w; // (L, C)
    floatX* ln1b; // (L, C)
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w; // (L, C)
    floatX* ln2b; // (L, C)
    floatX* fcw; // (L, 4*C, C)
    floatX* fcb; // (L, 4*C)
    floatX* fcprojw; // (L, C, 4*C)
    floatX* fcprojb; // (L, C)
    floatX* lnfw; // (C)
    floatX* lnfb; // (C)
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

// allocate memory for the parameters and point the individual tensors to the right places
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }
    void* params_memory = sycl::malloc_device(num_parameters_bytes, sycl_queue);
    if (params_memory == nullptr) { printf("SYCL malloc_device failed\n"); exit(EXIT_FAILURE); }
    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }
    return params_memory;
}

constexpr int NUM_ACTIVATION_TENSORS = 23;
typedef struct {
    floatX* encoded; // (B, T, C)
    floatX* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    floatX* atty; // (L, B, T, C)
    floatX* att; // (L, B, NH, T, T)
    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T), will be accumulated in micro-steps
    floatX* qkvr; // (L, B, T, 3*C)
    floatX* output; // scratch/logits buffer, (B, T, max(3C, NH*T, Vp))
    float* att_f32;         // fp32 attention staging (preatt/dpreatt), (B, NH, T, T)
    floatX* scratch_bt4c;   // (B, T, 4*C)
    floatX* scratch_btc;    // (B, T, C)
    float* gemm_scratch;    // fp32 GEMM staging, (B, T, Vp) — must not alias floatX buffers
} ActivationTensors;

struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};

#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, GPT2Config config, int recompute) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * T);
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(data->atty, L * B * T * C);
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T * T);
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * T * C);
    tensors[7] = TENSOR_SPEC(data->ln2, (recompute < 2) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * 4*C);
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * T * 4*C : B * T * 4*C);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkvr, L * B * T * 3*C);
    tensors[18] = TENSOR_SPEC(data->output, B * T * std::max((size_t)(3*C), std::max(NH*T, Vp)));
    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);
    tensors[21] = TENSOR_SPEC(data->gemm_scratch, B * T * Vp);
    tensors[22] = TENSOR_SPEC(data->att_f32, B * NH * T * T);
}

void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }
    printf0("allocating %d MiB for activations\n", (int)round(bytes / (1024 * 1024)));
    void* acts_memory = sycl::malloc_device(bytes, sycl_queue);
    if (acts_memory == nullptr) { printf("SYCL malloc_device failed\n"); exit(EXIT_FAILURE); }
    sycl_queue.memset(acts_memory, 0, bytes).wait();
    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        if (tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        } else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    ParameterTensors grads;
    void* grads_memory;
    float* m_memory;
    float* v_memory;
    float* master_weights;
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    int batch_size;
    int seq_len;
    int* inputs;
    int* targets;
    float mean_loss;
    float* accumulated_mean_loss;
    float* cpu_losses;
    unsigned long long rng_state;
    unsigned long long rng_state_last_update;
    int use_master_weights;
    bool init_state;
    int gelu_fusion;
    int recompute;
    int* workload_indices;
    int* bucket_info;
} GPT2;

void gpt2_init_common(GPT2 *model) {
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->accumulated_mean_loss = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
    model->params_memory = NULL;
    model->grads_memory = NULL;
    model->workload_indices = NULL;
    model->bucket_info = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->master_weights = NULL;
    model->rng_state = 13371337 + multi_gpu_config.process_rank;
    model->use_master_weights = 1;
    model->init_state = true;
    model->recompute = 1;
    model->gelu_fusion = 0;
}

void gpt2_allocate_weights(GPT2 *model) {
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

void gpt2_allocate_state(GPT2 *model, int B, int T) {
    printf0("allocating %d MiB for parameter gradients\n", (int)round(model->num_parameters * sizeof(floatX) / (1024 * 1024)));
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);

    model->batch_size = B;
    model->seq_len = T;

    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    model->inputs = sycl::malloc_device<int>((size_t)B * T, sycl_queue);
    model->targets = sycl::malloc_device<int>((size_t)B * T, sycl_queue);
    model->accumulated_mean_loss = sycl::malloc_device<float>(1, sycl_queue);
    model->cpu_losses = (float*)mallocCheck((size_t)B * T * sizeof(float));

    size_t xvec = (sizeof(floatX) == 2) ? 8 : 4;
    size_t num_c_groups = CEIL_DIV(model->config.channels, (32 * xvec));
    assert((size_t)(model->batch_size * model->seq_len) * num_c_groups < (1ULL<<31ULL));
    model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * model->seq_len * num_c_groups);
    model->bucket_info = (int*)mallocCheck(sizeof(int) * 4 * model->batch_size * model->seq_len * num_c_groups);

    int memory_status = 0;
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    printf0("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(float)) >> 20);
    printf0("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(float)) >> 20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    memory_status |= syclMallocConditionallyManaged((void**)&model->m_memory, shard_num_parameters * sizeof(float));

    memory_status |= syclMallocConditionallyManaged((void**)&model->v_memory, shard_num_parameters * sizeof(float));

    if (model->use_master_weights == 1) {
        assert(model->master_weights == nullptr);
        printf0("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        memory_status |= syclMallocConditionallyManaged((void**) &model->master_weights, shard_num_parameters * sizeof(float));
    }

    int reduced_memory_status = (int) multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced_memory_status >= 1) {
        printf0("WARNING: Fell back to shared memory when initializing m,v,master_weights\n");
    }
    printf0("device memory usage: unknown on SYCL device\n");
    size_t bytes_per_sequence = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes_per_sequence += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type) / B;
    }
    printf0("memory per sequence: %zu MiB\n", bytes_per_sequence / 1024 / 1024);
}

void gpt2_write_to_checkpoint(GPT2 *model, const char* checkpoint_path) {
    printf0("Writing model to %s\n", checkpoint_path);
    FILE *model_file = fopenCheck(checkpoint_path, "wb");
    int model_header[256];
    memset(model_header, 0, sizeof(model_header));
    model_header[0] = 20240326;
    assert(PRECISION_MODE == PRECISION_FP32 || PRECISION_MODE == PRECISION_BF16);
    model_header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5;
    model_header[2] = model->config.max_seq_len;
    model_header[3] = model->config.vocab_size;
    model_header[4] = model->config.num_layers;
    model_header[5] = model->config.num_heads;
    model_header[6] = model->config.channels;
    model_header[7] = model->config.padded_vocab_size;
    fwriteCheck(model_header, sizeof(int), 256, model_file);
    device_to_file(model_file, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, sycl_queue);
    fcloseCheck(model_file);
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, bool weight_init=true) {
    if (PRECISION_MODE == PRECISION_FP16) {
        fprintf(stderr, "build_from_checkpoint() does not support fp16 right now.\n");
        exit(EXIT_FAILURE);
    }
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    if (weight_init) {
        if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
            fprintf(stderr, "Precision is configured as BF16 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: are you sure you're loading a _bf16.bin file?\n");
            exit(EXIT_FAILURE);
        }
        if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
            fprintf(stderr, "Precision is configured as FP32 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: to turn on FP32 you have to compile with -DENABLE_FP32\n");
            fprintf(stderr, "---> HINT: are you sure you're loading a .bin file without any _bf16 in the name?\n");
            exit(EXIT_FAILURE);
        }
    }
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    gpt2_allocate_weights(model);

    if (weight_init) {
        assert(model->params_memory != NULL);
        file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, sycl_queue);
    }
    fcloseCheck(model_file);
    sycl_queue.wait();
}

void gpt2_set_hyperparameters(GPT2Config* config, const char* depth_str) {
    int depth = atoi(depth_str);
    assert(depth > 0);
    int channels, num_heads;
    if      (depth == 6)  { channels = 384; num_heads = 6; }
    else if (depth == 12) { channels = 768; num_heads = 12; }
    else if (depth == 24) { channels = 1024; num_heads = 16; }
    else if (depth == 36) { channels = 1280; num_heads = 20; }
    else if (depth == 48) { channels = 1600; num_heads = 25; }
    else if (depth == 60) { channels = 1920; num_heads = 30; }
    else if (depth == 72) { channels = 2880; num_heads = 30; }
    else if (depth == 84) { channels = 3456; num_heads = 36; }
    else { fprintf(stderr, "Unsupported GPT-2 depth: %d\n", depth); exit(EXIT_FAILURE); }
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = num_heads;
    config->max_seq_len = 1024;
}

void gpt3_set_hyperparameters(GPT2Config* config, const char* channels_str) {
    int channels = atoi(channels_str);
    assert(channels > 0);
    int depth, head_size;
    if      (channels == 384)   { depth = 6;  head_size = 64; }
    else if (channels == 768)   { depth = 12; head_size = 64; }
    else if (channels == 1024)  { depth = 24; head_size = 64; }
    else if (channels == 1536)  { depth = 24; head_size = 96; }
    else if (channels == 2048)  { depth = 24; head_size = 128; }
    else if (channels == 2560)  { depth = 32; head_size = 80; }
    else if (channels == 4096)  { depth = 32; head_size = 128; }
    else if (channels == 5140)  { depth = 40; head_size = 128; }
    else if (channels == 12288) { depth = 96; head_size = 128; }
    else { fprintf(stderr, "Unsupported GPT-3 channels: %d\n", channels); exit(EXIT_FAILURE); }
    assert(channels % head_size == 0);
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = channels / head_size;
    config->max_seq_len = 2048;
}

void gpt_build_from_descriptor(GPT2 *model, const char* descriptor) {
    assert(descriptor != NULL);
    size_t len = strlen(descriptor);
    if (len > 1 && descriptor[0] == 'd') {
        gpt2_set_hyperparameters(&model->config, descriptor + 1);
    } else if (len > 6 && strncmp(descriptor, "gpt2:d", 6) == 0) {
        gpt2_set_hyperparameters(&model->config, descriptor + 6);
    } else if (len > 6 && strncmp(descriptor, "gpt3:c", 6) == 0) {
        gpt3_set_hyperparameters(&model->config, descriptor + 6);
    } else {
        fprintf(stderr, "Unsupported model descriptor: %s\n", descriptor); exit(EXIT_FAILURE);
    }
    model->config.vocab_size = 50257;
    model->config.padded_vocab_size = 50304;

    gpt2_allocate_weights(model);

    mt19937_state init_rng;
    manual_seed(&init_rng, 42);
    floatX* params_memory_cpu = (floatX*)mallocCheck(model->num_parameters_bytes);
    memset(params_memory_cpu, 0, model->num_parameters_bytes);
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.num_layers);
    size_t L = model->config.num_layers;
    size_t offset = 0;
    for (size_t l = 0; l < L; l++) {
        offset = 0;
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            if (l == 0 && (i == 2 || i == 8 || i == 14)) {
                for (size_t j = 0; j < model->param_elements[i]; j++) {
                    params_memory_cpu[offset + j] = (floatX)1.0f;
                }
            }
            if ((l == 0 && (i == 0 || i == 1))
              || i == 4 || i == 6 || i == 10 || i == 12) {
                size_t n = model->param_elements[i];
                size_t layer_offset = 0;
                if (i == 0) {
                    n = model->config.vocab_size * model->config.channels;
                }
                if (i == 4 || i == 6 || i == 10 || i == 12) {
                    assert(n % L == 0);
                    n = n / L;
                    layer_offset = l * n;
                }
                float scale = (i == 6 || i == 12) ? 0.02f * residual_scale : 0.02f;
                float *fp32_buffer = (float*)mallocCheck(n * sizeof(float));
                normal_(fp32_buffer, n, 0.0f, scale, &init_rng);
                for (size_t j = 0; j < n; j++) {
                    params_memory_cpu[offset + layer_offset + j] = (floatX)fp32_buffer[j];
                }
                free(fp32_buffer);
            }
            offset += model->param_elements[i];
        }
    }
    sycl_queue.memcpy(model->params_memory, params_memory_cpu, model->num_parameters_bytes).wait();
    free(params_memory_cpu);
}

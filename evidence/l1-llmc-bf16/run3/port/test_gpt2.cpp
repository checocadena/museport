#define TESTING
#include "train_part1.cpp"
#include "train_part2.cpp"

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char* label, float threshold=1e-0) {
    int print_upto = 10;
    int ok = 1;
    float max_diff = 0.0f;
    float max_rel_error = 0.0f;
    float max_to_threshold = 0.f;
    float max_a = 0.0f;
    float max_b = 0.0f;
    float epsilon = 0.079;      // BF16 epsilon value
    printf("---\n");
    printf("checking tensor: %s\n", label);
    for (int i = 0; i < n; i++) {
        float t_eff = threshold + fabsf(b[i]) * epsilon;
        float diff = fabsf(a[i] - b[i]);
        max_to_threshold = std::max(max_to_threshold, diff / t_eff);
        if (diff > max_diff) {
            max_diff = diff;
            float denom = fabsf(b[i]);
            max_rel_error = (denom == 0.0f) ? 0.0f : diff / denom;
            max_a = a[i];
            max_b = b[i];
        }
        if (diff > t_eff) {
            ok = 0;
        }
        if (i < print_upto) {
            printf(diff <= t_eff ? "OK " :  "NOT OK ");
            printf("%f %f\n", a[i], b[i]);
        }
    }
    if (ok) {
        printf("TENSOR OK, max diff: %.3e, with rel error: %.3e (calculated=%10f, ref=%10f), %.2f%% of maximum error\n",
                max_diff, max_rel_error, max_a, max_b, max_to_threshold*100);
    } else {
        printf("TENSOR NOT OK, max diff: %.3e, with rel error: %.3e (calculated=%10f, ref=%10f), %.2f%% of maximum error\n",
                max_diff, max_rel_error, max_a, max_b, max_to_threshold*100);
    }
    return ok;
}

typedef struct {
    float*  wte;
    float*  wpe;
    float*  ln1w;
    float*  ln1b;
    float*  qkvw;
    float*  qkvb;
    float*  attprojw;
    float*  attprojb;
    float*  ln2w;
    float*  ln2b;
    float*  fcw;
    float*  fcb;
    float*  fcprojw;
    float*  fcprojb;
    float*  lnfw;
    float*  lnfb;
} FloatParameterTensors;
static_assert(sizeof(FloatParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

float* float_cpu_malloc_and_point_parameters(FloatParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

// main test (mirrors llm.c test_gpt2.cu exactly, with SYCL device transfers)
int main(int argc, char *argv[]) {
    char nccl_init_method[256] = "mpi";
    int num_processes = -1;
    int process_rank = -1;
    int gpus_per_node = -1;
    char server_ip[256] = "";
    char fs_path[256] = "";
    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    common_start(false, true);

    #if defined(ENABLE_BF16)
    const char* load_filename = "gpt2_124M_bf16.bin";
    #else
    const char* load_filename = "gpt2_124M.bin";
    #endif

    GPT2 model;
    gpt2_init_common(&model);

    gpt2_build_from_checkpoint(&model, load_filename);
    size_t V = model.config.vocab_size;
    size_t Vp = model.config.padded_vocab_size;
    size_t maxT = model.config.max_seq_len;

    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { exit(EXIT_FAILURE); }
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { exit(EXIT_FAILURE); }
        if (argv[i][0] != '-') { exit(EXIT_FAILURE); }
        if (argv[i][1] == 'w') { model.use_master_weights = atoi(argv[i+1]); }
        else if (argv[i][1] == 'r') { model.recompute = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { model.gelu_fusion = atoi(argv[i+1]); }
    }

    FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { fprintf(stderr, "Bad magic state file\n"); exit(EXIT_FAILURE); }
    if (state_header[1] != 2) {
        fprintf(stderr, "Bad version in state file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    int B = state_header[2];
    int T = state_header[3];
    assert(0 <= T && T <= (int)maxT);
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(x, sizeof(int), B*T, state_file);
    freadCheck(y, sizeof(int), B*T, state_file);
    float* expected_logits = (float*) mallocCheck(B * T * V * sizeof(float));
    float* expected_loss = (float*) mallocCheck(1 * sizeof(float));
    freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
    freadCheck(expected_loss, sizeof(float), 1, state_file);
    FloatParameterTensors expected_grads;
    float* expected_grads_memory = float_cpu_malloc_and_point_parameters(&expected_grads, model.param_elements);
    freadCheck(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
    fcloseCheck(state_file);

    void* grads_memory_cpu = mallocCheck(model.num_parameters_bytes);
    float* grads_memory_cpu_float = (float*)mallocCheck(model.num_parameters * sizeof(float));

    int allok = 1;

    gpt2_allocate_state(&model, B, T);

    gpt2_forward(&model, x, B, T);
    floatX* logits_cpu_raw = (floatX*)mallocCheck(B * T * Vp * sizeof(floatX));
    float* logits_cpu = (float*)mallocCheck(B * T * Vp * sizeof(float));
    sycl_queue.memcpy(logits_cpu_raw, model.acts.output, B * T * Vp * sizeof(floatX)).wait();
    for (int i = 0; i < B * T * (int)Vp; i++) {
        logits_cpu[i] = (float)logits_cpu_raw[i];
    }

    float logit_accuracy_threshold = 1e-3f;
    float loss_diff_threshold = 1e-5f;
    #if defined(ENABLE_BF16) || defined(ENABLE_F16)
    logit_accuracy_threshold = 25.0f;
    loss_diff_threshold = 0.05f;
    #endif

    int logits_ok = 1;
    float max_diff = 0.0f;
    for (int bt = 0; bt < B*T; bt++) {
        for (size_t v = 0; v < V; v++) {
            int i = bt * Vp + v;
            if (i < 10) {
                printf("%f, %f\n", expected_logits[i], logits_cpu[i]);
            }
            float diff = fabsf(expected_logits[bt*V + v] - logits_cpu[i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff >= logit_accuracy_threshold) {
                printf("MISMATCH AT INDEX %d,%d: ", bt, (int)v);
                printf("%f %f\n", expected_logits[bt*V + v], logits_cpu[i]);
                logits_ok = 0;
                bt = B*T;
                break;
            }
        }
    }
    allok = allok && logits_ok;
    if (!logits_ok) { printf("NOT "); }
    printf("OK (LOGITS)\n");
    printf("logit max diff: %f\n", max_diff);

    float losses[10];
    for (int step = 0; step < 10; step++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        gpt2_forward(&model, x, B, T);
        gpt2_backward_and_reduce(&model, x, y, 1, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        if (step == 0) {
            sycl_queue.memcpy(grads_memory_cpu, model.grads_memory, model.num_parameters_bytes).wait();

            char* src_iterator = (char*)grads_memory_cpu;
            float* dst_iterator = (float*)grads_memory_cpu_float;
            float* exp_iterator = expected_grads_memory;
            float* tensors1[NUM_PARAMETER_TENSORS];
            float* tensors2[NUM_PARAMETER_TENSORS];
            for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
                if (model.param_sizeof[i] == sizeof(float)) {
                    memcpy(dst_iterator, src_iterator, model.param_elements[i] * sizeof(float));
                } else {
                    assert(model.param_sizeof[i] == sizeof(floatX));
                    for (size_t j = 0; j < model.param_elements[i]; j++) {
                        dst_iterator[j] = (float)((floatX*)src_iterator)[j];
                    }
                }
                tensors1[i] = dst_iterator;
                tensors2[i] = exp_iterator;
                src_iterator += model.param_elements[i] * model.param_sizeof[i];
                dst_iterator += model.param_elements[i];
                exp_iterator += model.param_elements[i];
            }

            float grad_thresholds[NUM_PARAMETER_TENSORS] = {
                    5e-1f, 4e-3f, 1e-1f, 4e-2f,
                    5e-2f, 3.5e-2f, 2e-2f, 3e-2f,
                    5e-2f, 3e-2f, 3e-2f, 3e-2f,
                    2e-2f, 1e-2f,1e-1f,2e-2f};

            #if defined(ENABLE_FP32)
            for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
                grad_thresholds[i] = 1e-6f;
            }
            #endif
            const char* names[NUM_PARAMETER_TENSORS] = {
                    "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attrpojw",
                    "attprojb", "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb",
                    "lnfw", "lnfb"
            };
            size_t* count = model.param_elements;
            for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
                allok = allok & check_tensor(tensors1[i], tensors2[i], count[i], names[i], grad_thresholds[i]);
            }
        }

        float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
        float grad_scale = (grad_norm > 1.0f) ? 1.0f / grad_norm : 1.0f;
        gpt2_update(&model, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f, grad_scale, step+1, &multi_gpu_config);

        printf("step %d: loss %f (took %f ms)\n", step+1, model.mean_loss, time_elapsed_s * 1000);
        float rounded_loss = roundf(model.mean_loss * 1000000) / 1000000;
        losses[step] = rounded_loss;
    }

    float expected_losses[10] = {
        5.270009f,
        4.060681f,
        3.320085f,
        2.717550f,
        2.181066f,
        1.653923f,
        1.168050f,
        0.736873f,
        0.401021f,
        0.187493f
    };

    for (int i = 0; i < 10; i++) {
        if (fabsf(losses[i] - expected_losses[i]) >= loss_diff_threshold) {
            printf("LOSS MISMATCH AT STEP %d: %f %f\n", i+1, losses[i], expected_losses[i]);
            allok = 0;
        } else {
            printf("loss ok at step %d: %f %f\n", i+1, losses[i], expected_losses[i]);
        }
    }

    gpt2_write_to_checkpoint(&model, "test_gpt2cu_model.ckpt");

    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_val.bin", B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 1);
    save_state("test_gpt2cu_state.ckpt", 10, &model, &loader);
    int tokens[10];
    for (int step = 0; step < 10; step++) {
        dataloader_next_batch(&loader);
        gpt2_forward(&model, loader.inputs, B, T);
        gpt2_backward_and_reduce(&model, loader.inputs, loader.targets, 1, 0);
        gpt2_update(&model, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f, 1.0f, step+11, &multi_gpu_config);
        losses[step] = model.mean_loss;
        tokens[step] = loader.inputs[0];
    }

    gpt2_free(&model);
    gpt2_build_from_checkpoint(&model, "test_gpt2cu_model.ckpt");
    int ld_step;
    gpt2_allocate_state(&model, B, T);
    load_state(&ld_step, &model, &loader, "test_gpt2cu_state.ckpt");
    for (int step = 0; step < 10; step++) {
        dataloader_next_batch(&loader);
        gpt2_forward(&model, loader.inputs, B, T);
        gpt2_backward_and_reduce(&model, loader.inputs, loader.targets, 1, 0);
        gpt2_update(&model, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f, 1.0f, step+11, &multi_gpu_config);

        if (loader.inputs[0] != tokens[step]) {
            printf("Nondeterminism! Token mismatch at step %d: %d vs %d\n", step, tokens[step], loader.inputs[0]);
            allok = false;
            break;
        }

        if (losses[step] != model.mean_loss) {
            printf("Nondeterminism! Loss mismatch at step %d: %.15f vs %.15f\n", step, losses[step], model.mean_loss);
            allok = false;
            break;
        } else {
            printf("loss ok at step %d: %f %f\n", step, losses[step], model.mean_loss);
        }
    }

    printf("overall okay: %d\n", allok);

    remove("test_gpt2cu_model.ckpt");
    remove("test_gpt2cu_state.ckpt");

    dataloader_free(&loader);
    gpt2_free(&model);
    common_free(model);
    free(x);
    free(y);
    free(logits_cpu_raw);
    free(logits_cpu);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    free(grads_memory_cpu);
    free(grads_memory_cpu_float);
    return allok ? EXIT_SUCCESS : EXIT_FAILURE;
}

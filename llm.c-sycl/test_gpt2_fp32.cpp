#define TESTING
#include "train_gpt2_fp32.cpp"

// poor man's tensor checker
int check_tensor(float *a, float *b, size_t n, const char *label) {
    int print_upto = 5;
    int ok = 1;
    printf("%s\n", label);
    for (size_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) <= 1e-2f) {
            if ((int)i < print_upto) { printf("OK "); }
        } else {
            if ((int)i < print_upto) { printf("NOT OK "); }
            ok = 0;
        }
        if ((int)i < print_upto) { printf("%f %f\n", a[i], b[i]); }
    }
    if (ok) {
        printf("TENSOR OK\n");
    } else {
        printf("TENSOR NOT OK\n");
    }
    return ok;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    try {
        sycl::queue q{ sycl::property::queue::in_order{} };
        g_q = &q;
        std::string devname = q.get_device().get_info<sycl::info::device::name>();
        printf("[System]\n");
        printf("Device 0: %s\n", devname.c_str());

        // build the GPT-2 model from a checkpoint
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        int V = model.config.vocab_size;
        int Vp = model.config.padded_vocab_size;
        int maxT = model.config.max_seq_len;

        // load additional information that we will use for debugging and error checking
        FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
        int state_header[256];
        freadCheck(state_header, sizeof(int), 256, state_file);
        if (state_header[0] != 20240327) { printf("Bad magic state file\n"); exit(EXIT_FAILURE); }
        if (state_header[1] != 2) {
            fprintf(stderr, "Bad version in state file\n");
            fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
            exit(EXIT_FAILURE);
        }
        int B = state_header[2];
        int T = state_header[3];
        assert(0 <= T && T <= maxT);
        printf("[State]\n");
        printf("batch_size: %d\n", B);
        printf("seq_len: %d\n", T);

        float *expected_grads_memory = (float *)mallocCheck((size_t)model.num_parameters * sizeof(float));
        float *calculated_grads_memory = (float *)mallocCheck((size_t)model.num_parameters * sizeof(float));

        int *x = (int *)mallocCheck((size_t)B * T * sizeof(int));
        int *y = (int *)mallocCheck((size_t)B * T * sizeof(int));
        float *expected_logits = (float *)mallocCheck((size_t)B * T * V * sizeof(float));
        float *expected_loss = (float *)mallocCheck(1 * sizeof(float));

        freadCheck(x, sizeof(int), (size_t)B * T, state_file);
        freadCheck(y, sizeof(int), (size_t)B * T, state_file);
        freadCheck(expected_logits, sizeof(float), (size_t)B * T * V, state_file);
        freadCheck(expected_loss, sizeof(float), 1, state_file);
        freadCheck(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
        fcloseCheck(state_file);

        int allok = 1;

        // First, do target-free forward pass to validate logits
        gpt2_forward(&model, x, NULL, B, T);
        float *logits_cpu = (float *)mallocCheck((size_t)B * T * Vp * sizeof(float));
        qq().memcpy(logits_cpu, model.acts.output, (size_t)B * T * Vp * sizeof(float)).wait();

        int logits_ok = 1;
        for (int bt = 0; bt < B * T; bt++) {
            for (int v = 0; v < V; v++) {
                int i = bt * Vp + v;
                if (i < 10) {
                    printf("%f, %f\n", expected_logits[i], logits_cpu[i]);
                }
                float diff = fabsf(expected_logits[(size_t)bt * V + v] - logits_cpu[i]);
                if (diff >= 1e-2f) {
                    printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                    printf("%f %f\n", expected_logits[(size_t)bt * V + v], logits_cpu[i]);
                    logits_ok = 0;
                    bt = B * T;
                    break;
                }
            }
        }
        allok = allok && logits_ok;
        if (!logits_ok) { printf("NOT "); }
        printf("OK (LOGITS)\n");

        float losses[10];
        for (int step = 0; step < 10; step++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            gpt2_forward(&model, x, y, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

            if (step == 0) {
                free(logits_cpu);
                logits_cpu = NULL;

                if (fabsf(model.mean_loss - *expected_loss) >= 1e-2f) {
                    printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                    allok = 0;
                } else {
                    printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
                }

                qq().memcpy(calculated_grads_memory, model.grads_memory, model.num_parameters * sizeof(float)).wait();
                check_tensor(calculated_grads_memory, expected_grads_memory, model.num_parameters, "grads");
            }

            gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);

            printf("step %d: loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
            losses[step] = model.mean_loss;
        }

        float expected_losses[10] = {
            5.270007133483887f,
            4.059706687927246f,
            3.3751230239868164f,
            2.8007826805114746f,
            2.315382242202759f,
            1.8490285873413086f,
            1.3946564197540283f,
            0.9991465210914612f,
            0.6240804195404053f,
            0.37651097774505615f
        };

        for (int i = 0; i < 10; i++) {
            if (fabsf(losses[i] - expected_losses[i]) >= 1e-2f) {
                printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expected_losses[i]);
                allok = 0;
            } else {
                printf("loss ok at step %d: %f %f\n", i, losses[i], expected_losses[i]);
            }
        }

        printf("overall okay: %d\n", allok);

        free(x);
        free(y);
        free(expected_logits);
        free(expected_loss);
        free(expected_grads_memory);
        free(calculated_grads_memory);
        if (logits_cpu != NULL) { free(logits_cpu); }
        gpt2_free(&model);
    } catch (sycl::exception const &e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    return 0;
}

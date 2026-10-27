// Proof of the GPT-2 stage-3 check: llm.c's CPU train_gpt2.c with the same
// contract main.
#define TESTING
#include "train_gpt2.c"
#define MUSEPORT_SETUP
// museport stage 3 (training): the contract main. Runs 10 training
// steps on llm.c's debug batch as test_gpt2_fp32.cu does (forward with
// targets, zero the gradients, backward, AdamW update) and writes each step's
// mean loss as raw float32.
// usage: <prog> <debug_state.bin> <output> <gpt2_124M.bin>
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }
    MUSEPORT_SETUP
    GPT2 model;
    gpt2_build_from_checkpoint(&model, argv[3]);
    int sh[256];
    FILE* s = fopenCheck(argv[1], "rb");
    freadCheck(sh, sizeof(int), 256, s);
    int B = sh[2], T = sh[3];
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(x, sizeof(int), B * T, s);
    freadCheck(y, sizeof(int), B * T, s);
    fcloseCheck(s);
    float losses[10];
    for (int step = 0; step < 10; step++) {
        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        losses[step] = model.mean_loss;
    }
    FILE* o = fopenCheck(argv[2], "wb");
    fwrite(losses, sizeof(float), 10, o);
    fcloseCheck(o);
    free(x);
    free(y);
    gpt2_free(&model);
    return 0;
}

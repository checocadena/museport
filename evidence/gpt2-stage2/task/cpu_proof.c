// Proof of the GPT-2 stage-2 check: llm.c's CPU train_gpt2.c with the same
// contract main.
#define TESTING
#include "train_gpt2.c"
#define MUSEPORT_SETUP
#define MUSEPORT_COPY_GRADS(dst, n) memcpy(dst, model.grads_memory, (n) * sizeof(float))
// museport stage 2 (backward): the contract main. Runs a forward pass
// with targets on llm.c's debug batch, then the backward pass, and writes raw
// float32: the gradient of every parameter, then the mean loss.
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
    gpt2_forward(&model, x, y, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    size_t P = model.num_parameters;
    float* grads = (float*)mallocCheck(P * sizeof(float));
    MUSEPORT_COPY_GRADS(grads, P);
    FILE* o = fopenCheck(argv[2], "wb");
    fwrite(grads, sizeof(float), P, o);
    float loss = model.mean_loss;
    fwrite(&loss, sizeof(float), 1, o);
    fcloseCheck(o);
    free(x);
    free(y);
    free(grads);
    gpt2_free(&model);
    return 0;
}

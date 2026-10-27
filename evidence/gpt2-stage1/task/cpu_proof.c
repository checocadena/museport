// Proof of the GPT-2 stage-1 check: llm.c's CPU train_gpt2.c with the same contract main.
#define TESTING
#include "train_gpt2.c"
#define MUSEPORT_SETUP
#define MUSEPORT_COPY_LOGITS(dst, n) memcpy(dst, model.acts.logits, (n) * sizeof(float))
// museport stage 1 (forward): the contract main. Runs the model on llm.c's debug batch and
// writes raw float32: the B*T*V logits of a target-free forward pass, then the mean loss.
// usage: <prog> <debug_state.bin> <output> <gpt2_124M.bin>
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }
    MUSEPORT_SETUP
    GPT2 model;
    gpt2_build_from_checkpoint(&model, argv[3]);
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int sh[256];
    FILE* s = fopenCheck(argv[1], "rb");
    freadCheck(sh, sizeof(int), 256, s);
    int B = sh[2], T = sh[3];
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(x, sizeof(int), B * T, s);
    freadCheck(y, sizeof(int), B * T, s);
    fcloseCheck(s);
    float* logits = (float*)mallocCheck((size_t)B * T * Vp * sizeof(float));
    gpt2_forward(&model, x, NULL, B, T);
    MUSEPORT_COPY_LOGITS(logits, (size_t)B * T * Vp);
    gpt2_forward(&model, x, y, B, T);
    FILE* o = fopenCheck(argv[2], "wb");
    for (int bt = 0; bt < B * T; bt++) fwrite(logits + (size_t)bt * Vp, sizeof(float), V, o);
    float loss = model.mean_loss;
    fwrite(&loss, sizeof(float), 1, o);
    fcloseCheck(o);
    free(x);
    free(y);
    free(logits);
    gpt2_free(&model);
    return 0;
}

// Hidden reference for GPT-2 stage 2 (backward): PyTorch's gradients of
// every parameter and the loss for llm.c's debug batch, read from
// gpt2_124M_debug_state.bin, written as raw float32: the gradients, then the loss.
// usage: reference <debug_state.bin> <output> <gpt2_124M.bin>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void must(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}
int main(int argc, char** argv) {
    must(argc == 4, "usage: reference <debug_state.bin> <output> <gpt2_124M.bin>");
    int mh[256], sh[256];
    FILE* m = std::fopen(argv[3], "rb");
    must(m && std::fread(mh, 4, 256, m) == 256, "weights header");
    std::fclose(m);
    must(mh[0] == 20240326, "bad weights magic");
    FILE* s = std::fopen(argv[1], "rb");
    must(s && std::fread(sh, 4, 256, s) == 256, "state header");
    must(sh[0] == 20240327 && sh[1] == 2, "bad state magic or version");
    const long B = sh[2], T = sh[3], V = mh[3];
    // header, x, y, logits, then the loss; the gradients fill the rest of the file
    const long loss_at = 4L * (256 + 2 * B * T + B * T * V);
    must(std::fseek(s, 0, SEEK_END) == 0, "seek");
    const long end = std::ftell(s);
    const long P = (end - loss_at - 4) / 4;
    must(P > 0 && loss_at + 4 + 4 * P == end, "state file size");
    float loss = 0.0f;
    must(std::fseek(s, loss_at, SEEK_SET) == 0 && std::fread(&loss, 4, 1, s) == 1, "loss");
    std::vector<float> grads(P);
    must(std::fread(grads.data(), 4, P, s) == (size_t)P, "gradients");
    std::fclose(s);
    FILE* o = std::fopen(argv[2], "wb");
    must(o && std::fwrite(grads.data(), 4, P, o) == (size_t)P, "write");
    must(std::fwrite(&loss, 4, 1, o) == 1, "write loss");
    std::fclose(o);
    return 0;
}

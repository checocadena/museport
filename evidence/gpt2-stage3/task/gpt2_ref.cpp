// Hidden reference for GPT-2 stage 3 (training): PyTorch's loss at each of
// 10 AdamW steps on llm.c's debug batch, as llm.c's test_gpt2_fp32.cu lists
// them, written as raw float32.
// usage: reference <debug_state.bin> <output> <gpt2_124M.bin>
#include <cstdio>
#include <cstdlib>
static void must(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}
int main(int argc, char** argv) {
    must(argc == 4, "usage: reference <debug_state.bin> <output> <gpt2_124M.bin>");
    int sh[256];
    FILE* s = std::fopen(argv[1], "rb");
    must(s && std::fread(sh, 4, 256, s) == 256, "state header");
    must(sh[0] == 20240327 && sh[1] == 2, "bad state magic or version");
    std::fclose(s);
    const float losses[] = {5.270007133483887f, 4.059706687927246f, 3.3751230239868164f, 2.8007826805114746f, 2.315382242202759f, 1.8490285873413086f, 1.3946564197540283f, 0.9991465210914612f, 0.6240804195404053f, 0.37651097774505615f};
    FILE* o = std::fopen(argv[2], "wb");
    const size_t n = sizeof(losses) / sizeof(losses[0]);
    must(o && std::fwrite(losses, 4, n, o) == n, "write");
    std::fclose(o);
    return 0;
}

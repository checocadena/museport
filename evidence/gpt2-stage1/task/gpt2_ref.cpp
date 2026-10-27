// Hidden reference for GPT-2 stage 1 (forward): PyTorch's logits and loss for llm.c's debug
// batch, read from gpt2_124M_debug_state.bin, written as raw float32: B*T*V logits, then the
// loss. usage: reference <debug_state.bin> <output> <gpt2_124M.bin>
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
    std::vector<int> xy(2 * B * T);
    must(std::fread(xy.data(), 4, xy.size(), s) == xy.size(), "x, y");
    std::vector<float> out(B * T * V + 1);
    must(std::fread(out.data(), 4, out.size(), s) == out.size(), "logits, loss");
    std::fclose(s);
    FILE* o = std::fopen(argv[2], "wb");
    must(o && std::fwrite(out.data(), 4, out.size(), o) == out.size(), "write");
    std::fclose(o);
    return 0;
}

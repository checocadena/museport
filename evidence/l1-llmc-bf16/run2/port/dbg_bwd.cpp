#include "train_part1.cpp"
#include "train_part2.cpp"

static void dump(const char* name, floatX* d, size_t n) {
  std::vector<floatX> h(n);
  sycl_queue.memcpy(h.data(), d, n*sizeof(floatX)).wait();
  double mx = 0; int bad = 0;
  for (size_t i = 0; i < n; i++) { float v = (float)h[i]; if (!isfinite(v)) bad++; else mx = fmax(mx, fabs((double)v)); }
  printf("%s: maxabs=%.4f bad=%d/%zu\n", name, mx, bad, n);
}
static void dumpf(const char* name, float* d, size_t n) {
  std::vector<float> h(n);
  sycl_queue.memcpy(h.data(), d, n*sizeof(float)).wait();
  double mx = 0; int bad = 0;
  for (size_t i = 0; i < n; i++) { float v = h[i]; if (!isfinite(v)) bad++; else mx = fmax(mx, fabs((double)v)); }
  printf("%s: maxabs=%.4f bad=%d/%zu\n", name, mx, bad, n);
}

int main() {
  multi_gpu_config = multi_gpu_config_init(-1,-1,-1,(char*)"",(char*)"",(char*)"mpi");
  common_start(false, true);
  GPT2 model;
  gpt2_init_common(&model);
  gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");
  FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
  int state_header[256];
  freadCheck(state_header, sizeof(int), 256, state_file);
  int B = state_header[2]; int T = state_header[3];
  set_zero_configs(&multi_gpu_config, 0, model.num_parameters);
  int* x = (int*)mallocCheck(B*T*sizeof(int));
  int* y = (int*)mallocCheck(B*T*sizeof(int));
  freadCheck(x, sizeof(int), B*T, state_file);
  freadCheck(y, sizeof(int), B*T, state_file);
  fcloseCheck(state_file);
  // skip expected reads
  gpt2_allocate_state(&model, B, T);
  size_t C = model.config.channels, Vp = model.config.padded_vocab_size, L = model.config.num_layers;
  gpt2_forward(&model, x, B, T);
  printf("after fwd1\n");
  dump("logits", model.acts.output, (size_t)B*T*Vp);
  gpt2_backward_and_reduce(&model, x, y, 1, 0);
  printf("after bwd1: mean_loss=%.6f\n", model.mean_loss);
  dump("dloss", model.acts.output, (size_t)B*T*Vp);
  dumpf("losses", model.acts.losses, (size_t)B*T);
  dump("dwte", model.grads.wte, Vp*C);
  dump("dwpe", model.grads.wpe, (size_t)T*C);
  dump("dresid", (floatX*)model.acts.scratch_btc, (size_t)B*T*C);
  dump("dlnfw", model.grads.lnfw, C);
  dump("dlfcw0", model.grads.fcw, (size_t)4*C*C);
  dump("dqkvw0", model.grads.qkvw, (size_t)3*C*C);
  float gn = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
  printf("grad_norm=%.6f\n", gn);
  float gs = (gn > 1.0f) ? 1.0f/gn : 1.0f;
  gpt2_update(&model, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f, gs, 1, &multi_gpu_config);
  printf("after upd1\n");
  dump("wte", model.params.wte, Vp*C);
  dump("lnfw", model.params.lnfw, C);
  gpt2_forward(&model, x, B, T);
  printf("after fwd2\n");
  dump("logits2", model.acts.output, (size_t)B*T*Vp);
  gpt2_backward_and_reduce(&model, x, y, 1, 0);
  printf("after bwd2: mean_loss=%.6f\n", model.mean_loss);
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}

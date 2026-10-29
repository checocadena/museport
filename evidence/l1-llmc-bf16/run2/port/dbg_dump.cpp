#define DEBUG_DUMP
#include "train_part1.cpp"
#include "train_part2.cpp"

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
  gpt2_allocate_state(&model, B, T);
  size_t C = model.config.channels, Vp = model.config.padded_vocab_size;
  gpt2_forward(&model, x, B, T);
  gpt2_backward_and_reduce(&model, x, y, 1, 0);
  printf("mean_loss=%.6f\n", model.mean_loss);
  // isolate: re-run encoder_backward on fresh buffers with captured inputs
  {
    // capture dresidual by re-running full backward? Instead: use current grads state.
    // Fresh dwte2/dwpe2 zeroed, call wte path via encoder_backward with model.inputs
    size_t nres = (size_t)B*T*C;
    std::vector<floatX> hres(nres);
    // dresidual buffer was scratch_btc; after backward it still holds dresidual (encoder only reads it)
    sycl_queue.memcpy(hres.data(), model.acts.scratch_btc, nres*sizeof(floatX)).wait();
    floatX* d_res = sycl::malloc_device<floatX>(nres, sycl_queue);
    floatX* d_w2 = sycl::malloc_device<floatX>(Vp*C, sycl_queue);
    floatX* d_p2 = sycl::malloc_device<floatX>((size_t)T*C, sycl_queue);
    floatX* d_sc = sycl::malloc_device<floatX>((size_t)B*T*3*C, sycl_queue);
    int* d_in = sycl::malloc_device<int>((size_t)B*T, sycl_queue);
    sycl_queue.memcpy(d_res, hres.data(), nres*sizeof(floatX));
    sycl_queue.memset(d_w2, 0, Vp*C*sizeof(floatX));
    sycl_queue.memset(d_p2, 0, (size_t)T*C*sizeof(floatX));
    sycl_queue.memcpy(d_in, x, (size_t)B*T*sizeof(int)).wait();
    // need device copy of model->inputs? encoder_backward takes dout, inp(device), inputs_cpu(host)
    encoder_backward(d_w2, d_p2, d_sc, model.workload_indices, model.bucket_info,
                     d_res, d_in, x, B, T, (int)C, 999, sycl_queue);
    std::vector<floatX> hw(Vp*C);
    sycl_queue.memcpy(hw.data(), d_w2, Vp*C*sizeof(floatX)).wait();
    int bad = 0;
    for (size_t i = 0; i < Vp*C; i++) if (!isfinite((float)hw[i])) bad++;
    printf("isolated wte rerun: bad=%d/%zu\n", bad, Vp*C);
  }
  // locate bad dwte elements
  {
    size_t n = Vp*C;
    std::vector<floatX> h(n);
    sycl_queue.memcpy(h.data(), model.grads.wte, n*sizeof(floatX)).wait();
    int shown = 0, bad = 0;
    for (size_t i = 0; i < n; i++) {
      float v = (float)h[i];
      if (!isfinite(v)) {
        bad++;
        if (shown < 20) { printf("bad dwte[%zu] row=%zu col=%zu val=%f\n", i, i/C, i%C, v); shown++; }
      }
    }
    printf("total bad=%d\n", bad);
    // distinct input tokens
    int seen[50304] = {0};
    for (int i = 0; i < B*T; i++) seen[x[i]] = 1;
    int nd = 0; for (int i = 0; i < 50304; i++) nd += seen[i];
    printf("distinct tokens=%d\n", nd);
  }
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}

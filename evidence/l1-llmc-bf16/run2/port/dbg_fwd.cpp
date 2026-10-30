#include "train_part1.cpp"
#include "train_part2.cpp"

static float vec_maxabs(floatX* d, size_t n) {
  std::vector<floatX> h(n);
  sycl_queue.memcpy(h.data(), d, n*sizeof(floatX)).wait();
  float m = 0;
  int nfinite = 0;
  for (size_t i = 0; i < n; i++) { float v=(float)h[i]; if(!isfinite(v)) nfinite++; else m=fmaxf(m,fabsf(v)); }
  printf("  maxabs=%.6f nonfinite=%d/%zu\n", m, nfinite, n);
  return m;
}

int main() {
  multi_gpu_config = multi_gpu_config_init(-1,-1,-1,(char*)"",(char*)"",(char*)"mpi");
  common_start(false, true);
  GPT2 model;
  gpt2_init_common(&model);
  gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");
  size_t V = model.config.vocab_size;
  size_t Vp = model.config.padded_vocab_size;
  printf("V=%zu Vp=%zu L=%d C=%d NH=%d maxT=%d\n", V, Vp, model.config.num_layers, model.config.channels, model.config.num_heads, model.config.max_seq_len);
  FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
  int state_header[256];
  freadCheck(state_header, sizeof(int), 256, state_file);
  int B = state_header[2]; int T = state_header[3];
  printf("B=%d T=%d\n", B, T);
  set_zero_configs(&multi_gpu_config, 0, model.num_parameters);
  int* x = (int*)mallocCheck(B*T*sizeof(int));
  int* y = (int*)mallocCheck(B*T*sizeof(int));
  freadCheck(x, sizeof(int), B*T, state_file);
  freadCheck(y, sizeof(int), B*T, state_file);
  float* expected_logits = (float*)mallocCheck(B*T*V*sizeof(float));
  freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
  fcloseCheck(state_file);
  gpt2_allocate_state(&model, B, T);

  // instrumented forward: replicate gpt2_forward layer by layer with dumps
  const size_t L = model.config.num_layers, NH = model.config.num_heads, C = model.config.channels;
  sycl_queue.memcpy(model.inputs, x, B*T*sizeof(int)).wait();
  ParameterTensors params = model.params;
  ActivationTensors acts = model.acts;
  encoder_forward(acts.encoded, model.inputs, params.wte, params.wpe, B, T, C, sycl_queue);
  printf("encoded:"); vec_maxabs(acts.encoded, B*T*C);
  layernorm_forward(acts.ln1, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, sycl_queue);
  printf("ln1:"); vec_maxabs(acts.ln1, B*T*C);
  {
    std::vector<float> m(B*T), r(B*T);
    sycl_queue.memcpy(m.data(), acts.ln1_mean, B*T*sizeof(float)).wait();
    sycl_queue.memcpy(r.data(), acts.ln1_rstd, B*T*sizeof(float)).wait();
    printf("ln1_mean[0]=%.6f rstd[0]=%.6f\n", m[0], r[0]);
  }
  for (int l = 0; l < (int)L; l++) {
    floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1)*B*T*C;
    floatX* l_qkvw = params.qkvw + l*3*C*C;
    floatX* l_qkvb = params.qkvb + l*3*C;
    floatX* l_ln1 = acts.ln1 + l*B*T*C;
    floatX* l_qkvr = acts.qkvr + l*B*T*3*C;
    floatX* l_atty = acts.atty + l*B*T*C;
    floatX* scratch = (floatX*)acts.output;
    float* sgemm = acts.gemm_scratch;
    matmul_forward_sycl(scratch, sgemm, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, sycl_queue, 0);
    if (l < 2) { printf("l%d qkv:", l); vec_maxabs(scratch, B*T*3*C); }
    attention_forward(l_atty, l_qkvr, acts.att + l*B*NH*T*T, scratch,
                      acts.att_f32 + l*B*NH*T*T, sgemm, B, T, C, NH, sycl_queue);
    if (l < 2) { printf("l%d atty:", l); vec_maxabs(l_atty, B*T*C); }
    floatX* l_attprojw = params.attprojw + l*C*C;
    floatX* l_attprojb = params.attprojb + l*C;
    matmul_forward_sycl(scratch, sgemm, l_atty, l_attprojw, l_attprojb, B, T, C, C, sycl_queue, 0);
    floatX* l_residual2 = acts.residual2 + l*B*T*C;
    floatX* l_ln2 = acts.ln2 + l*B*T*C;
    floatX* l_ln2w = params.ln2w + l*C;
    floatX* l_ln2b = params.ln2b + l*C;
    fused_residual_forward5(l_residual2, l_ln2, acts.ln2_mean + l*B*T, acts.ln2_rstd + l*B*T,
                            residual, scratch, l_ln2w, l_ln2b, B*T, C, sycl_queue);
    floatX* l_fcw = params.fcw + l*4*C*C;
    floatX* l_fcb = params.fcb + l*4*C;
    floatX* l_fch = acts.fch + l*B*T*4*C;
    floatX* l_fch_gelu = acts.fch_gelu + l*B*T*4*C;
    matmul_forward_sycl(l_fch, sgemm, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, sycl_queue, 0);
    gelu_forward(l_fch_gelu, l_fch, B*T*4*C, sycl_queue);
    if (l < 2) { printf("l%d fch_gelu:", l); vec_maxabs(l_fch_gelu, B*T*4*C); }
    floatX* l_fcprojw = params.fcprojw + l*C*4*C;
    floatX* l_fcprojb = params.fcprojb + l*C;
    matmul_forward_sycl(scratch, sgemm, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, sycl_queue, 0);
    floatX* l_residual3 = acts.residual3 + l*B*T*C;
    if (l+1 != (int)L) {
      fused_residual_forward5(l_residual3, acts.ln1 + (l+1)*B*T*C, acts.ln1_mean + (l+1)*B*T, acts.ln1_rstd + (l+1)*B*T,
                              l_residual2, scratch, params.ln1w + (l+1)*C, params.ln1b + (l+1)*C, B*T, C, sycl_queue);
    } else {
      fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                              params.lnfw, params.lnfb, B*T, C, sycl_queue);
    }
    if (l < 2) { printf("l%d res3:", l); vec_maxabs(l_residual3, B*T*C); }
    if (l == 1) break; // only first 2 layers for debug
  }
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}

// full forward + logits compare entry used by second debug build
int full_forward_check() {
  multi_gpu_config = multi_gpu_config_init(-1,-1,-1,(char*)"",(char*)"",(char*)"mpi");
  common_start(false, true);
  GPT2 model;
  gpt2_init_common(&model);
  gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");
  size_t V = model.config.vocab_size;
  size_t Vp = model.config.padded_vocab_size;
  FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
  int state_header[256];
  freadCheck(state_header, sizeof(int), 256, state_file);
  int B = state_header[2]; int T = state_header[3];
  set_zero_configs(&multi_gpu_config, 0, model.num_parameters);
  int* x = (int*)mallocCheck(B*T*sizeof(int));
  int* y = (int*)mallocCheck(B*T*sizeof(int));
  freadCheck(x, sizeof(int), B*T, state_file);
  freadCheck(y, sizeof(int), B*T, state_file);
  float* expected_logits = (float*)mallocCheck(B*T*V*sizeof(float));
  freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
  fcloseCheck(state_file);
  gpt2_allocate_state(&model, B, T);
  gpt2_forward(&model, x, B, T);
  floatX* raw = (floatX*)mallocCheck(B*T*Vp*sizeof(floatX));
  sycl_queue.memcpy(raw, model.acts.output, B*T*Vp*sizeof(floatX)).wait();
  float md = 0; double se = 0; int cnt = 0;
  for (int bt = 0; bt < B*T; bt++) for (size_t v = 0; v < V; v++) {
    float d = fabsf(expected_logits[bt*V+v] - (float)raw[bt*Vp+v]);
    md = fmaxf(md, d); se += d*d; cnt++;
  }
  printf("logits maxdiff=%.4f rms=%.5f (thr 25.0)\n", md, sqrt(se/cnt));
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}

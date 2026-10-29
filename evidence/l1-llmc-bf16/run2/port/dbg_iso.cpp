#define DEBUG_DUMP
#include "train_part1.cpp"
#include "train_part2.cpp"

static int count_bad(floatX* d, size_t n) {
  std::vector<floatX> h(n);
  sycl_queue.memcpy(h.data(), d, n*sizeof(floatX)).wait();
  int bad = 0;
  for (size_t i = 0; i < n; i++) if (!isfinite((float)h[i])) bad++;
  return bad;
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
  gpt2_allocate_state(&model, B, T);
  size_t V = model.config.vocab_size, Vp = model.config.padded_vocab_size;
  size_t L = model.config.num_layers, NH = model.config.num_heads, C = model.config.channels;
  ParameterTensors params = model.params;
  ParameterTensors grads = model.grads;
  ActivationTensors acts = model.acts;
  gpt2_forward(&model, x, B, T);

  // replicate backward up to layer 11's attention, with sentinel on grads.wte
  const float dloss = 1.0f / (float)(B * T * 1);
  sycl_queue.memcpy(model.targets, y, B * T * sizeof(int)).wait();
  fused_classifier(acts.output, acts.losses, dloss, model.targets, B, T, V, Vp, True, sycl_queue);
  floatX* dresidual = (floatX*)model.acts.scratch_btc;
  sycl_queue.memset(dresidual, 0, B * T * C * sizeof(floatX)).wait();
  sycl_queue.memset(model.grads_memory, 0, model.num_parameters * sizeof(floatX)).wait();
  float* scratchF = (float*)acts.output;
  floatX* scratchX = (floatX*)acts.output;
  matmul_backward_sycl(model.acts.scratch_bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp, sycl_queue);
  // sentinel grads.wte to 1.0
  {
    std::vector<floatX> one(Vp*C, (floatX)1.0f);
    // copy in chunks to avoid big host buffer? 38M*2=77MB fine
    sycl_queue.memcpy(grads.wte, one.data(), Vp*C*sizeof(floatX)).wait();
  }
  floatX* residual = acts.residual3 + (L-1) * B * T * C;
  layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, model.acts.scratch_bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, sycl_queue);
  printf("sentinel set; bad=%d\n", count_bad(grads.wte, Vp*C));
  int l = 11;
  floatX* l_ln2w = params.ln2w + l * C;
  floatX* l_ln2b = params.ln2b + l * C;
  floatX* l_fcw = params.fcw + l * 4*C * C;
  floatX* l_fcb = params.fcb + l * 4*C;
  floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
  floatX* l_fcprojb = params.fcprojb + l * C;
  floatX* l_attprojw = params.attprojw + l * C * C;
  floatX* l_attprojb = params.attprojb + l * C;
  floatX* dl_attprojw = grads.attprojw + l * C * C;
  floatX* dl_attprojb = grads.attprojb + l * C;
  floatX* dl_ln2w = grads.ln2w + l * C;
  floatX* dl_ln2b = grads.ln2b + l * C;
  floatX* dl_fcw = grads.fcw + l * 4*C * C;
  floatX* dl_fcb = grads.fcb + l * 4*C;
  floatX* dl_fcprojw = grads.fcprojw + l * C * 4*C;
  floatX* dl_fcprojb = grads.fcprojb + l * C;
  floatX* l_residual2 = acts.residual2 + l * B * T * C;
  floatX* l_ln2 = acts.ln2 + l * B * T * C;
  float* l_ln2_mean = acts.ln2_mean + l * B * T;
  float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
  floatX* l_fch_pre_gelu = acts.fch + l * B * T * 4*C;
  floatX* l_fch_gelu = acts.fch_gelu;
  floatX* l_atty = acts.atty + l * B * T * C;
  floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
  floatX* l_att = acts.att + l * B * NH * T * T;
  floatX* dl_btc = residual;
  floatX* dl_bt4c = (floatX*)model.acts.scratch_bt4c;
  gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, sycl_queue);
  matmul_backward_sycl(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C, sycl_queue);
  gelu_backward_inplace(dl_bt4c, l_fch_pre_gelu, B*T*4*C, sycl_queue);
  printf("after fcproj: bad=%d\n", count_bad(grads.wte, Vp*C));
  matmul_backward_sycl(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C, sycl_queue);
  layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, sycl_queue);
  matmul_backward_sycl(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C, sycl_queue);
  printf("after mlp: bad=%d\n", count_bad(grads.wte, Vp*C));
  // inline attention_backward with per-substep checks
  {
    floatX* buffer_a = l_atty;
    floatX* buffer_b = l_fch_pre_gelu;
    const int HS = C / NH;
    const floatX *qq = l_qkvr + 0 * B * T * C;
    const floatX *kk = l_qkvr + 1 * B * T * C;
    const floatX *vv = l_qkvr + 2 * B * T * C;
    floatX *dq = buffer_b + 0 * B * T * C;
    floatX *dk = buffer_b + 1 * B * T * C;
    floatX *dv = buffer_b + 2 * B * T * C;
    floatX* dinp = dl_bt4c;
    floatX* datt = scratchX;
    floatX* scratch = buffer_a;
    float* dpreattF = acts.att_f32 + l * B * NH * T * T;
    float* tmpF = acts.gemm_scratch;
    const floatX* dout = dl_btc;
    unpermute_backward(scratch, dout, B, T, NH, HS, sycl_queue);
    printf("after unpermute: bad=%d\n", count_bad(grads.wte, Vp*C));
#ifndef SKIP_DATT_GEMM
    sycl_gemm_batch_datt(dpreattF, vv, scratch, T, HS, (int64_t)B * NH, sycl_queue);
#endif
    printf("after datt-gemm: bad=%d\n", count_bad(grads.wte, Vp*C));
    { // convert
      size_t N = (size_t)B * NH * T * T;
      sycl_queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        datt[id[0]] = (floatX)dpreattF[id[0]];
      }).wait();
    }
    printf("after datt-convert: bad=%d\n", count_bad(grads.wte, Vp*C));
    sycl_gemm_batch_dv(tmpF, scratch, l_att, T, HS, (int64_t)B * NH, sycl_queue);
    printf("after dv-gemm: bad=%d\n", count_bad(grads.wte, Vp*C));
    { // convert dv
      size_t N = (size_t)B * NH * T * HS;
      sycl_queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        dv[id[0]] = (floatX)tmpF[id[0]];
      }).wait();
    }
    printf("after dv-convert: bad=%d\n", count_bad(grads.wte, Vp*C));
    softmax_backward_sycl(datt, l_att, B * NH, T, 1.0f/sqrtf((float)HS), sycl_queue);
    printf("after softmax-bwd: bad=%d\n", count_bad(grads.wte, Vp*C));
    sycl_gemm_batch_dq(tmpF, kk, datt, T, HS, (int64_t)B * NH, sycl_queue);
    printf("after dq-gemm: bad=%d\n", count_bad(grads.wte, Vp*C));
    sycl_gemm_batch_dk(tmpF, qq, datt, T, HS, (int64_t)B * NH, sycl_queue);
    printf("after dk-gemm: bad=%d\n", count_bad(grads.wte, Vp*C));
    permute_qkv_backward(dinp, dq, dk, dv, B, T, NH, HS, sycl_queue);
    printf("after permute-bwd: bad=%d\n", count_bad(grads.wte, Vp*C));
  }
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}

// Part 2: forward / validate / backward / grad norm / update / free / common / state

#ifdef DEBUG_DUMP
#define DBG_COUNT_BAD(ptr, n, label) do { \
    size_t _dbg_n = (size_t)(n); \
    std::vector<floatX> _dbg_h(_dbg_n); \
    sycl_queue.memcpy(_dbg_h.data(), (const void*)(ptr), _dbg_n*sizeof(floatX)).wait(); \
    int _dbg_bad = 0; size_t _dbg_first = 0; \
    for (size_t _dbg_i = 0; _dbg_i < _dbg_n; _dbg_i++) if (!isfinite((float)_dbg_h[_dbg_i])) { if (_dbg_bad==0) _dbg_first=_dbg_i; _dbg_bad++; } \
    printf("DBG %s: bad=%d/%zu first=%zu\n", (label), _dbg_bad, _dbg_n, _dbg_first); \
} while(0)
#else
#define DBG_COUNT_BAD(ptr, n, label) do {} while(0)
#endif

// propagate inputs through the network to produce logits.
void gpt2_forward(GPT2 *model, const int* inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    if (B > (size_t)model->batch_size || T > (size_t)model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }

    sycl_queue.memcpy(model->inputs, inputs, B * T * sizeof(int)).wait();
    tokenCheck(inputs, B*T, V);

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, sycl_queue);

    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd,
                      acts.encoded, params.ln1w, params.ln1b, B, T, C, sycl_queue);

    for (int l = 0; l < (int)L; l++) {
        SyclRange layer_range("Layer", l);

        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch = (floatX*)acts.output;
        // fp32 GEMM staging lives in a dedicated buffer (must not alias floatX data)
        float* sgemm = acts.gemm_scratch;

        floatX* l_att = acts.att + l * B * NH * T * T;
        if (T != (size_t)model->seq_len) {
            sycl_queue.memset(l_att, 0, B * NH * T * T * sizeof(floatX)).wait();
        }
        // qkv matmul: scratch(BT x 3C) = l_ln1(BT x C) * l_qkvw(3C x C)^T + bias
        // NOTE: sgemm aliases scratch memory; sycl_gemm_forward fully writes the
        // fp32 staging before the epilogue reads it, so this is safe.
        matmul_forward_sycl(scratch, sgemm, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, sycl_queue, 0);
        // attention fp32 staging: preatt in att_f32 slice, vaccum in gemm_scratch slice
        // (gemm_scratch holds B*T*Vp floats; vaccum needs B*T*C <= that)
        attention_forward(l_atty, l_qkvr, l_att, scratch,
                          acts.att_f32 + l * B * NH * T * T, sgemm,
                          B, T, C, NH, sycl_queue);

        matmul_forward_sycl(scratch, sgemm, l_atty, l_attprojw, l_attprojb, B, T, C, C, sycl_queue, 0);
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, sycl_queue);
        if (model->gelu_fusion >= 1) {
            matmul_forward_gelu_sycl(l_fch_gelu, l_fch, sgemm, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, sycl_queue, 0);
        } else {
            matmul_forward_sycl(l_fch, sgemm, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, sycl_queue, 0);
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C, sycl_queue);
        }
        matmul_forward_sycl(scratch, sgemm, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, sycl_queue, 0);
        if (l+1 != (int)L) {
            floatX* l_ln1n = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, l_ln1n, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * T, C, sycl_queue);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B * T, C, sycl_queue);
        }
    }

    {
        // classifier: fp32 staging in dedicated buffer (must not alias logits)
        float* sf = acts.gemm_scratch;
        matmul_forward_sycl(acts.output, sf, acts.lnf, params.wte, NULL, B, T, C, Vp, sycl_queue, 0);
    }
    sycl_queue.wait();
    (void)V;
}

float gpt2_validate(GPT2 *model, const int* inputs, const int* targets, size_t B, size_t T) {
    assert(targets != NULL);
    gpt2_forward(model, inputs, B, T);
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;

    SyclRange classifier_and_loss_range("classifier_and_loss");
    ActivationTensors acts = model->acts;
    float mean_loss = 0.0f;
    const float dloss = 1.0f / (B * T);
    sycl_queue.memset(acts.losses, 0, B*T*sizeof(float)).wait();
    sycl_queue.memcpy(model->targets, targets, B * T * sizeof(int)).wait();
    tokenCheck(targets, B*T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, False, sycl_queue);
    sycl_queue.memcpy(model->cpu_losses, acts.losses, B * T * sizeof(float)).wait();
    for (size_t i = 0; i < B*T; i++) {
        mean_loss += model->cpu_losses[i];
    }
    mean_loss /= B*T;
    sycl_queue.wait();
    return mean_loss;
}

void gpt2_backward_and_reduce(GPT2 *model, int* inputs, const int* targets, int grad_accum_steps, int micro_step) {
    if (model->grads_memory == nullptr) {
        fprintf(stderr, "Need to allocate gradients before backward");
        exit(EXIT_FAILURE);
    }
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    if (micro_step == 0) {
        sycl_queue.memset(model->acts.losses, 0, (size_t)model->batch_size * model->seq_len * sizeof(float)).wait();
        sycl_queue.memset(model->grads_memory, 0, model->num_parameters * sizeof(floatX)).wait();
    }

    const size_t B = model->batch_size;
    const size_t T = model->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;

    SyclRange classifier_and_loss_range("classifier_and_loss");
    const float dloss = 1.0f / (float)(B * T * grad_accum_steps);
    sycl_queue.memcpy(model->targets, targets, B * T * sizeof(int)).wait();
    tokenCheck(targets, B*T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, True, sycl_queue);
    DBG_COUNT_BAD((const void*)acts.output, B*T*Vp, "dlogits");

    floatX* dresidual = (floatX*)model->acts.scratch_btc;
    sycl_queue.memset(dresidual, 0, B * T * C * sizeof(floatX)).wait();

    float*  scratchF = (float*)acts.output;
    floatX* scratchX = (floatX*)acts.output;

    // classifier matmul backward: dlogits(BT x Vp) is in acts.output
    matmul_backward_sycl(model->acts.scratch_bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp, sycl_queue);
    DBG_COUNT_BAD((const void*)grads.wte, Vp*C, "dwte-after-classifier");
    floatX* residual = acts.residual3 + (L-1) * B * T * C;
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, model->acts.scratch_bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, sycl_queue);
    DBG_COUNT_BAD((const void*)grads.wte, Vp*C, "dwte-after-lnfbwd");

    floatX* dl_btc = residual;

    for (int l = (int)L-1; l >= 0; l--) {
        SyclRange layer_range("Layer", l);

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_ln1b = params.ln1b + l * C;
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* dl_ln1w = grads.ln1w + l * C;
        floatX* dl_ln1b = grads.ln1b + l * C;
        floatX* dl_qkvw = grads.qkvw + l * 3*C * C;
        floatX* dl_qkvb = grads.qkvb + l * 3*C;
        floatX* dl_attprojw = grads.attprojw + l * C * C;
        floatX* dl_attprojb = grads.attprojb + l * C;
        floatX* dl_ln2w = grads.ln2w + l * C;
        floatX* dl_ln2b = grads.ln2b + l * C;
        floatX* dl_fcw = grads.fcw + l * 4*C * C;
        floatX* dl_fcb = grads.fcb + l * 4*C;
        floatX* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        floatX* dl_fcprojb = grads.fcprojb + l * C;
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch_pre_gelu = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;

        floatX* dl_bt4c = (floatX*)model->acts.scratch_bt4c;

        if (model->recompute >= 1) {
            gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, sycl_queue);
        }
        // fcproj backward with gelu fusion flag: pass pre_gelu only if fusion>=2, else do gelu backward separately
        if (model->gelu_fusion >= 2) {
            matmul_backward_sycl(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C, sycl_queue, l_fch_pre_gelu);
        } else {
            matmul_backward_sycl(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C, sycl_queue);
            if (model->gelu_fusion < 2 && true) {
                gelu_backward_inplace(dl_bt4c, l_fch_pre_gelu, B*T*4*C, sycl_queue);
            }
        }
        if (l == 11) { DBG_COUNT_BAD((const void*)grads.wte, Vp*C, "dwte-l11-fcproj"); }
        if (model->recompute >= 2) {
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, sycl_queue);
        }
        matmul_backward_sycl(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C, sycl_queue);
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, sycl_queue);
        matmul_backward_sycl(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C, sycl_queue);
        if (l == 11) { DBG_COUNT_BAD((const void*)grads.wte, Vp*C, "dwte-l11-mlpdone"); }

        floatX* l_att = acts.att + l * B * NH * T * T;
        floatX* buffer_a = l_atty;
        floatX* buffer_b = l_fch_pre_gelu;
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a,
                           acts.att_f32 + l * B * NH * T * T, acts.gemm_scratch,
                           dl_btc, l_qkvr, l_att, B, T, C, NH, sycl_queue);
        if (l == 11) { DBG_COUNT_BAD((const void*)grads.wte, Vp*C, "dwte-l11-attn"); }
        if (model->recompute >= 2) {
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, sycl_queue);
        }
        matmul_backward_sycl(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C, sycl_queue);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, sycl_queue);
        DBG_COUNT_BAD((const void*)grads.wte, Vp*C, "dwte-in-loop");

        if (last_step) {
            floatX* const pointers[] = {
                dl_ln1w, dl_ln1b,
                dl_qkvw, dl_qkvb,
                dl_attprojw, dl_attprojb,
                dl_ln2w, dl_ln2b,
                dl_fcw, dl_fcb,
                dl_fcprojw, dl_fcprojb
            };
            const size_t nelem[] = {
                C, C,
                3 * C * C, 3 * C,
                C * C, C,
                C, C,
                4 * C * C, 4 * C,
                C * 4 * C, C
            };
            multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, sycl_queue);
        }
    }
    encoder_backward(grads.wte, grads.wpe, scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), sycl_queue);
    DBG_COUNT_BAD((const void*)grads.wte, Vp*C, "dwte-after-encoder");

    if (last_step) {
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B*T, sycl_queue);
        sycl_queue.memcpy(&model->mean_loss, model->accumulated_mean_loss, sizeof(float)).wait();
        floatX* const pointers[] = {grads.wte, grads.wpe, grads.lnfw, grads.lnfb};
        const size_t nelem[] = {Vp * C, (size_t)T * C, C, C};
        multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, sycl_queue);
    }

    sycl_queue.wait();
    if (last_step) {
        model->mean_loss /= B*T*grad_accum_steps;
    } else {
        model->mean_loss = -1.f;
    }
}

ShardInfo gpt2_get_tensor_at_layer(const GPT2 *model, int layer_id, int param_tensor_id) {
    ptrdiff_t offset = 0;
    for (int i = 0; i < param_tensor_id; i++) {
        offset += (ptrdiff_t)model->param_elements[i];
    }
    size_t size = model->param_elements[param_tensor_id];
    if (2 <= param_tensor_id && param_tensor_id <= 13) {
        size /= model->config.num_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

float gpt2_calculate_grad_norm(GPT2 *model, MultiGpuConfig* multi_gpu_config) {
    NVTX_RANGE_FN();
    floatX* grads_memory = (floatX*)model->grads_memory;
    float* grad_norm_squared = (float*)model->acts.output;
    float grad_norm_squared_cpu = 0.0f;

    int num_slices[2] = {1, model->config.num_layers};
    int max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    if (multi_gpu_config->zero_stage == 1) {
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
            ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
            ptrdiff_t offset = tensor.offset + shard.offset;
            bool is_first_pass = (i == 0);
            if ((i < 2 || i > 13)) {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, 0, 1,
                                    max_num_block_sums, is_first_pass, sycl_queue);
            } else {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, tensor.size, model->config.num_layers,
                                    max_num_block_sums, is_first_pass, sycl_queue);
            }
        }
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, sycl_queue);
    } else {
        global_norm_squared(grad_norm_squared, grads_memory, model->num_parameters, 0, 1, max_num_block_sums, true, sycl_queue);
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, sycl_queue);
    }
    sycl_queue.memcpy(&grad_norm_squared_cpu, grad_norm_squared, sizeof(float)).wait();
    float grad_norm_cpu = sqrtf(grad_norm_squared_cpu);
    return grad_norm_cpu;
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, float grad_scale, int t,
                 MultiGpuConfig* multi_gpu_config, bool init_from_master_only=false) {
    NVTX_RANGE_FN();
    if (model->grads_memory == nullptr || model->m_memory == nullptr || model->v_memory == nullptr) {
        fprintf(stderr, "Need to allocate optimizer state before update");
        exit(EXIT_FAILURE);
    }

    bool init_state = model->init_state;
    if (init_state) {
        model->init_state = false;
        SyclRange rng("InitOpt");
        sycl_queue.memset(model->m_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(float)).wait();
        sycl_queue.memset(model->v_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(float)).wait();
    }

    model->rng_state_last_update = model->rng_state;

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        unsigned int seed = random_u32(&model->rng_state);

        int num_layers = model->config.num_layers;
        if ((i < 2 || i > 13)) {
            num_layers = 1;
        }

        ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
        ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
        ptrdiff_t local_offset_full = tensor.offset + shard.offset;
        ptrdiff_t local_offset_partial = tensor.offset / multi_gpu_config->num_processes;

        float wd = (i == 0 || i == 1 || i == 4 || i == 6 || i == 10 || i == 12) ? weight_decay : 0.0f;
        floatX* param_ptr = (floatX*)model->params_memory + local_offset_full;
        floatX* grad_ptr = (floatX*)model->grads_memory + local_offset_full;

        ptrdiff_t opt_state_offset = multi_gpu_config->zero_stage < 1 ? local_offset_full : local_offset_partial;
        float* m_ptr = model->m_memory + opt_state_offset;
        float* v_ptr = model->v_memory + opt_state_offset;
        float* master_ptr = nullptr;
        if (model->master_weights != nullptr) { master_ptr = model->master_weights + opt_state_offset; }
        if (init_state && model->master_weights != nullptr) {
            copy_and_cast_kernel(master_ptr, param_ptr, shard.size, shard.size, tensor.size, num_layers, sycl_queue);
        }

        if (init_from_master_only) {
            init_from_master(param_ptr, master_ptr, shard.size, tensor.size, shard.size, num_layers, seed, sycl_queue);
        } else {
            adamw_update(param_ptr, master_ptr, grad_ptr,
                         m_ptr, v_ptr,
                         shard.size, tensor.size, tensor.size, shard.size, num_layers,
                         learning_rate,
                         beta1, beta2, t, eps, wd, grad_scale, seed, sycl_queue);
        }
    }
    sycl_queue.wait();
}

float gpt2_estimate_mfu(GPT2 *model, int num_tokens, float dt) {
    size_t N = model->num_parameters;
    int L = model->config.num_layers;
    int C = model->config.channels;
    int T = model->seq_len;
    size_t flops_per_token = 6 * N + (size_t)6 * L * C * T;
    size_t flops_per_step = flops_per_token * num_tokens;
    float flops_achieved = (float)flops_per_step * (1.0f / dt);
    float flops_promised = get_flops_promised(sycl_device_name.c_str(), PRECISION_MODE) * 1e12f;
    if (flops_promised < 0) {
        return -1.f;
    }
    float mfu = flops_achieved / flops_promised;
    return mfu;
}

void gpt2_free(GPT2 *model) {
    syclFreeCheck(&model->params_memory);
    syclFreeCheck(&model->grads_memory);
    syclFreeCheck(&model->m_memory);
    syclFreeCheck(&model->v_memory);
    syclFreeCheck(&model->master_weights);
    syclFreeCheck(&model->acts_memory);
    syclFreeCheck(&model->inputs);
    syclFreeCheck(&model->targets);
    syclFreeCheck(&model->accumulated_mean_loss);
    free(model->cpu_losses);
    model->cpu_losses = NULL;
    free(model->workload_indices);
    model->workload_indices = NULL;
    free(model->bucket_info);
    model->bucket_info = NULL;
}

void common_start(bool override_enable_tf32 = true, bool print_device_info = true) {
    (void)override_enable_tf32;
    try {
        sycl_queue = sycl::queue(sycl::default_selector_v);
    } catch (sycl::exception& e) {
        printf("[SYCL ERROR] no device: %s\n", e.what());
        exit(EXIT_FAILURE);
    }
    sycl_device_name = sycl_queue.get_device().get_info<sycl::info::device::name>();
    if (print_device_info) {
        printf("[System]\n");
        printf("Device: %s\n", sycl_device_name.c_str());
    }
}

void common_free(GPT2 &model) {
    (void)model;
}

void save_state(const char* filename, int step, GPT2* model, DataLoader* loader) {
    printf("Writing state to %s\n", filename);
    FILE *state_file = fopenCheck(filename, "wb");
    int state_header[256];
    memset(state_header, 0, sizeof(state_header));
    state_header[0] = 20240527;
    state_header[1] = 1;
    state_header[2] = multi_gpu_config.num_processes;
    state_header[3] = multi_gpu_config.process_rank;
    state_header[4] = model->use_master_weights;
    state_header[5] = loader->should_shuffle;
    state_header[10] = step;
    *((unsigned long long*)&state_header[20]) = model->rng_state;
    *((unsigned long long*)&state_header[22]) = model->rng_state_last_update;
    *((size_t*)&state_header[30]) = loader->current_shard_idx;
    *((size_t*)&state_header[32]) = loader->current_sample_idx;
    fwriteCheck(state_header, sizeof(int), 256, state_file);

    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    device_to_file(state_file, model->m_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, sycl_queue);
    device_to_file(state_file, model->v_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, sycl_queue);
    if (model->use_master_weights) {
        device_to_file(state_file, model->master_weights, shard_num_parameters * sizeof(float), IO_BUF_SIZE, sycl_queue);
    }
    if (loader->should_shuffle) {
        fwriteCheck(&loader->glob_result.gl_pathc, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        fwriteCheck(&loader->shard_num_samples, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        fwriteCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    fcloseCheck(state_file);
}

void load_state(int* step, GPT2* model, DataLoader* loader, const char* filename) {
    FILE *state_file = fopenCheck(filename, "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    assert(state_header[0] == 20240527);
    assert(state_header[1] == 1);
    assert(state_header[2] == multi_gpu_config.num_processes);
    assert(state_header[3] == multi_gpu_config.process_rank);
    int use_master_weights = state_header[4];
    int should_shuffle = state_header[5];
    *step = state_header[10];
    model->rng_state = *((unsigned long long*)&state_header[20]);
    model->rng_state_last_update = *((unsigned long long*)&state_header[22]);
    size_t current_shard_idx = *((size_t*)&state_header[30]);
    size_t current_sample_idx = *((size_t*)&state_header[32]);

    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    if (use_master_weights == 1 && !model->use_master_weights) {
        printf0("Warning: Master weights are present in state, but not enabled for current run.");
    } else if (use_master_weights == 0 && model->use_master_weights) {
        printf0("Error: Master weights requested, but not present in state file.");
        exit(EXIT_FAILURE);
    }

    model->init_state = false;
    assert(model->m_memory != nullptr);
    assert(model->v_memory != nullptr);
    file_to_device(model->m_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, sycl_queue);
    file_to_device(model->v_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, sycl_queue);
    if (model->use_master_weights) {
        assert(model->master_weights != nullptr);
        file_to_device(model->master_weights, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, sycl_queue);
        model->rng_state = model->rng_state_last_update;
        gpt2_update(model, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, &multi_gpu_config, true);
        model->rng_state = *((unsigned long long*)&state_header[20]);
    }

    loader->should_shuffle = should_shuffle;
    if (should_shuffle == 1) {
        size_t glob_result_gl_pathc;
        freadCheck(&glob_result_gl_pathc, sizeof(size_t), 1, state_file);
        assert(glob_result_gl_pathc == loader->glob_result.gl_pathc);
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        freadCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        size_t shard_num_samples;
        freadCheck(&shard_num_samples, sizeof(size_t), 1, state_file);
        assert(shard_num_samples == loader->shard_num_samples);
        loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
        freadCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        freadCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    dataloader_resume(loader, current_shard_idx, current_sample_idx);
    fcloseCheck(state_file);
}

void write_checkpoint(const char* output_log_dir, int step, GPT2* model, DataLoader* train_loader, MultiGpuConfig* multi_gpu_config) {
    printf0("Writing checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        gpt2_write_to_checkpoint(model, filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    save_state(filename_buffer, step, model, train_loader);
    multi_gpu_barrier(multi_gpu_config);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        FILE* done_file = fopenCheck(filename_buffer, "w");
        fcloseCheck(done_file);
    }
}

void delete_checkpoint(const char* output_log_dir, int step, MultiGpuConfig* multi_gpu_config) {
    printf0("Deleting checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        remove(filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    remove(filename_buffer);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        remove(filename_buffer);
    }
}

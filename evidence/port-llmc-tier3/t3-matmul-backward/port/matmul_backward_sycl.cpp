#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Naive bias kernel: one work-item per output channel, sequential
// double-precision accumulation over B*T, exactly like the CUDA naive kernel.
void matmul_backward_bias_kernel_naive_sycl(sycl::queue &q, float *dbias,
                                            const float *dout, int B, int T,
                                            int OC) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(OC)),
                   [=](sycl::id<1> idx) {
                     int o = static_cast<int>(idx[0]);
                     double sum = 0.0;
                     for (int b = 0; b < B; ++b) {
                       for (int t = 0; t < T; ++t) {
                         sum += dout[(std::size_t)b * (std::size_t)T * (std::size_t)OC +
                                     (std::size_t)t * (std::size_t)OC +
                                     (std::size_t)o];
                       }
                     }
                     dbias[o] = static_cast<float>(sum);
                   });
  });
}

// Faster bias kernel: one work-group per output channel, thread coarsening
// with a local-memory tree reduction. 1-D launch, local-space barriers.
void matmul_backward_bias_kernel_faster_sycl(sycl::queue &q, float *dbias,
                                             const float *dout, int B, int T,
                                             int OC) {
  if (OC <= 0) {
    return;
  }
  std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  std::size_t local = 256;
  if (local > max_wg) {
    local = 1;
    while (local * 2 <= max_wg) {
      local *= 2;
    }
    if (local == 0) {
      local = 1;
    }
  }
  sycl::range<1> local_range(local);
  sycl::range<1> global_range(static_cast<std::size_t>(OC) * local);
  int n = B * T;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> shared(sycl::range<1>(local), h);
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                   [=](sycl::nd_item<1> it) {
                     int o = static_cast<int>(it.get_group(0));
                     int tid = static_cast<int>(it.get_local_id(0));
                     int block_size =
                         static_cast<int>(it.get_local_range(0));
                     const float *x = dout + o;
                     double sum = 0.0;
                     for (int i = tid; i < n; i += block_size) {
                       sum += x[(std::size_t)i * (std::size_t)OC];
                     }
                     shared[tid] = static_cast<float>(sum);
                     sycl::group_barrier(it.get_group());
                     for (int stride = block_size / 2; stride >= 1;
                          stride /= 2) {
                       sycl::group_barrier(it.get_group());
                       if (tid < stride) {
                         shared[tid] += shared[tid + stride];
                       }
                     }
                     if (tid == 0) {
                       dbias[o] = shared[0];
                     }
                   });
  });
}

// version 1: oneMath GEMMs + faster bias kernel, beta=1 accumulation
void matmul_backward1(sycl::queue &q, float *dinp, float *dweight, float *dbias,
                      const float *dout, const float *inp, const float *weight,
                      const float *ones, int B, int T, int C, int OC) {
  (void)ones;
  std::int64_t bt = static_cast<std::int64_t>(B) * static_cast<std::int64_t>(T);
  std::int64_t cll = static_cast<std::int64_t>(C);
  std::int64_t ocll = static_cast<std::int64_t>(OC);
  float alpha = 1.0f;
  float beta = 1.0f;
  // backward to input: dinp(C x BT) = weight(C x OC) * dout(OC x BT)
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::nontrans,
      oneapi::math::transpose::nontrans, cll, bt, ocll, alpha, weight, cll,
      dout, ocll, beta, dinp, cll)
      .wait();
  // backward to weight: dweight(C x OC) = inp(C x BT) * dout^T(BT x OC)
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
      cll, ocll, bt, alpha, inp, cll, dout, ocll, beta, dweight, cll)
      .wait();
  if (dbias != nullptr) {
    matmul_backward_bias_kernel_faster_sycl(q, dbias, dout, B, T, OC);
    q.wait();
  }
}

void matmul_backward(sycl::queue &q, int kernel_num, float *dinp,
                     float *dweight, float *dbias, const float *dout,
                     const float *inp, const float *weight, const float *ones,
                     int B, int T, int C, int OC) {
  switch (kernel_num) {
  case 1:
    matmul_backward1(q, dinp, dweight, dbias, dout, inp, weight, ones, B, T, C,
                     OC);
    break;
  default:
    std::printf("Invalid kernel number\n");
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C OC\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int OC = std::atoi(argv[6]);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *dout =
      (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * OC));
  if (std::fread(dout, sizeof(float), (std::size_t)(B * T * OC), fin) !=
      (std::size_t)(B * T * OC)) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  float *inp =
      (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * C));
  if (std::fread(inp, sizeof(float), (std::size_t)(B * T * C), fin) !=
      (std::size_t)(B * T * C)) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *weight =
      (float *)std::malloc(sizeof(float) * (std::size_t)(OC * C));
  if (std::fread(weight, sizeof(float), (std::size_t)(OC * C), fin) !=
      (std::size_t)(OC * C)) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  float *ones = (float *)std::malloc(sizeof(float) * (std::size_t)(OC));
  if (std::fread(ones, sizeof(float), (std::size_t)(OC), fin) !=
      (std::size_t)(OC)) {
    std::fprintf(stderr, "input too short at ones\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  try {
    float *d_dout = sycl::malloc_device<float>((std::size_t)(B * T * OC), q);
    float *d_inp = sycl::malloc_device<float>((std::size_t)(B * T * C), q);
    float *d_weight = sycl::malloc_device<float>((std::size_t)(OC * C), q);
    float *d_ones = sycl::malloc_device<float>((std::size_t)(OC), q);
    float *d_dinp = sycl::malloc_device<float>((std::size_t)(B * T * C), q);
    float *d_dweight = sycl::malloc_device<float>((std::size_t)(OC * C), q);
    float *d_dbias = sycl::malloc_device<float>((std::size_t)(OC), q);
    if (!d_dout || !d_inp || !d_weight || !d_ones || !d_dinp || !d_dweight ||
        !d_dbias) {
      std::fprintf(stderr, "device allocation failed\n");
      return 1;
    }

    q.memcpy(d_dout, dout, sizeof(float) * (std::size_t)(B * T * OC)).wait();
    q.memcpy(d_inp, inp, sizeof(float) * (std::size_t)(B * T * C)).wait();
    q.memcpy(d_weight, weight, sizeof(float) * (std::size_t)(OC * C)).wait();
    q.memcpy(d_ones, ones, sizeof(float) * (std::size_t)(OC)).wait();

    float *dinp = (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * C));
    float *dweight =
        (float *)std::malloc(sizeof(float) * (std::size_t)(OC * C));
    float *dbias = (float *)std::malloc(sizeof(float) * (std::size_t)(OC));

    const int variants[] = {1};
    for (int vi = 0; vi < 1; ++vi) {
      const int k = variants[vi];
      q.memset(d_dinp, 0, sizeof(float) * (std::size_t)(B * T * C)).wait();
      q.memset(d_dweight, 0, sizeof(float) * (std::size_t)(OC * C)).wait();
      q.memset(d_dbias, 0, sizeof(float) * (std::size_t)(OC)).wait();
      matmul_backward(q, k, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight,
                      d_ones, B, T, C, OC);
      q.wait();
      q.memcpy(dinp, d_dinp, sizeof(float) * (std::size_t)(B * T * C)).wait();
      std::fwrite(dinp, sizeof(float), (std::size_t)(B * T * C), fout);
      q.memcpy(dweight, d_dweight, sizeof(float) * (std::size_t)(OC * C))
          .wait();
      std::fwrite(dweight, sizeof(float), (std::size_t)(OC * C), fout);
      q.memcpy(dbias, d_dbias, sizeof(float) * (std::size_t)(OC)).wait();
      std::fwrite(dbias, sizeof(float), (std::size_t)(OC), fout);
    }
    std::fclose(fout);

    sycl::free(d_dout, q);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_ones, q);
    sycl::free(d_dinp, q);
    sycl::free(d_dweight, q);
    sycl::free(d_dbias, q);
    std::free(dout);
    std::free(inp);
    std::free(weight);
    std::free(ones);
    std::free(dinp);
    std::free(dweight);
    std::free(dbias);
  } catch (sycl::exception const &e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
  }
  return 0;
}

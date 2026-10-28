#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using floatX = float;
constexpr int kVec = 4; // fp32 Packed128 width

inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }
inline bool is_pow2(int n) { return n > 0 && ((n & (n - 1)) == 0); }
inline int largest_pow2_le(int n) {
  if (n < 1) return 0;
  while ((n & (n - 1)) > 0) n = n & (n - 1);
  return n;
}

// distinct kernel names for each variant
class BiasK1;
class BiasK2;
class BiasK3;
class BiasK4;
class BiasK5;
class BiasK7;
class BiasK7Tail;
class BiasK8;
class BiasK9;
class BiasK9Reduce;

void run_k1(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC, int block_size) {
  int L = largest_pow2_le(block_size);
  if (L < 1) L = 1;
  int G = OC;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> loc(sycl::range<1>(L), h);
     h.parallel_for<BiasK1>(
         sycl::nd_range<1>(sycl::range<1>(G * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           int o = (int)it.get_group(0);
           int tid = (int)it.get_local_id(0);
           int BT = B * T;
           const float *x = dout + o;
           float sum = 0.0f;
           for (int i = tid; i < BT; i += L) sum += x[i * OC];
           loc[tid] = sum;
           sycl::group_barrier(it.get_group());
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) loc[tid] += loc[tid + s];
             sycl::group_barrier(it.get_group());
           }
           if (tid == 0) dbias[o] = dbias[o] + loc[0];
         });
   }).wait();
}

void run_k2(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC) {
  constexpr int L = 32;
  int G = OC;
  q.submit([&](sycl::handler &h) {
     h.parallel_for<BiasK2>(
         sycl::nd_range<1>(sycl::range<1>(G * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           auto grp = it.get_group();
           int idx = (int)it.get_group(0);
           int lid = (int)it.get_local_id(0);
           int BT = B * T;
           float sum = 0.0f;
           for (int i = lid; i < BT; i += L) sum += dout[i * OC + idx];
           float gsum =
               sycl::reduce_over_group(grp, sum, sycl::plus<float>{});
           if (lid == 0) dbias[idx] = dbias[idx] + gsum;
         });
   }).wait();
}

void run_k3(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC, int block_size) {
  int L = block_size;
  int G = OC;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> loc(sycl::range<1>(32), h);
     h.parallel_for<BiasK3>(
         sycl::nd_range<1>(sycl::range<1>(G * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           int idx = (int)it.get_group(0);
           int lid = (int)it.get_local_id(0);
           int BT = B * T;
           float tsum = 0.0f;
           for (int i = lid; i < BT; i += L) tsum += dout[i * OC + idx];
           sycl::sub_group sg = it.get_sub_group();
           float sg_sum =
               sycl::reduce_over_group(sg, tsum, sycl::plus<float>{});
           if (sg.get_local_linear_id() == 0)
             loc[sg.get_group_linear_id()] = sg_sum;
           sycl::group_barrier(it.get_group());
           if (lid == 0) {
             uint32_t nsg = sg.get_group_linear_range();
             if (nsg > 32) nsg = 32;
             float bsum = 0.0f;
             for (uint32_t u = 0; u < nsg; ++u) bsum += loc[u];
             dbias[idx] = dbias[idx] + bsum;
           }
         });
   }).wait();
}

void run_k4(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC, int block_size) {
  constexpr int kTile = 32;
  int L = block_size;
  int vstep = L / kTile;
  int G = OC / kTile;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> sm(sycl::range<1>(L), h);
     h.parallel_for<BiasK4>(
         sycl::nd_range<1>(sycl::range<1>(G * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           int lid = (int)it.get_local_id(0);
           int gid = (int)it.get_group(0);
           int warp_id = lid / kTile;
           int lane = lid % kTile;
           int tl = gid * kTile;
           int BT = B * T;
           const float *col = dout + tl + lane;
           float s = 0.0f;
           for (int row = warp_id; row < BT; row += vstep)
             s += col[row * OC];
           sm[lane + warp_id * kTile] = s;
           sycl::group_barrier(it.get_group());
           if (warp_id == 0) {
             float acc = 0.0f;
             for (int j = 0; j < vstep; ++j) acc += sm[lane + j * kTile];
             dbias[tl + lane] = dbias[tl + lane] + acc;
           }
         });
   }).wait();
}

void run_k5(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC, int block_size) {
  int L = block_size;
  int Gx = ceil_div_int(OC, L);
  constexpr int Gy = 4;
  int G = Gx * Gy;
  q.submit([&](sycl::handler &h) {
     h.parallel_for<BiasK5>(
         sycl::nd_range<1>(sycl::range<1>(G * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           int gid = (int)it.get_group(0);
           int lid = (int)it.get_local_id(0);
           int gx = gid % Gx;
           int gy = gid / Gx;
           int oc = gx * L + lid;
           if (oc >= OC) return;
           int BT = B * T;
           float sum = 0.0f;
           for (int row = gy; row < BT; row += Gy) sum += dout[row * OC + oc];
           sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
               ar(dbias[oc]);
           ar.fetch_add(sum);
         });
   }).wait();
}

// k7: tiled Vec=4, tile 128 per group, Ty=8 rows share OCs via local scratch
void run_k7(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC) {
  constexpr int Tx = 32;
  constexpr int Ty = 8;
  constexpr int L = Tx * Ty; // 256
  constexpr int Tile = Tx * kVec; // 128
  int Gx = ceil_div_int(OC, Tile);
  int BT = B * T;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> scratch(sycl::range<1>(Ty * Tile), h);
     h.parallel_for<BiasK7>(
         sycl::nd_range<1>(sycl::range<1>(Gx * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           int gx = (int)it.get_group(0);
           int lid = (int)it.get_local_id(0);
           int tx = lid % Tx;
           int ty = lid / Tx;
           int base = gx * Tile + tx * kVec;
           float acc[kVec];
           for (int k = 0; k < kVec; ++k) acc[k] = 0.0f;
           for (int row = ty; row < BT; row += Ty) {
             for (int k = 0; k < kVec; ++k) {
               int c = base + k;
               if (c < OC) acc[k] += dout[row * OC + c];
             }
           }
           for (int k = 0; k < kVec; ++k)
             scratch[ty * Tile + tx * kVec + k] = acc[k];
           sycl::group_barrier(it.get_group());
           if (ty == 0) {
             for (int k = 0; k < kVec; ++k) {
               int c = base + k;
               if (c >= OC) continue;
               float s = 0.0f;
               for (int r = 0; r < Ty; ++r)
                 s += scratch[r * Tile + tx * kVec + k];
               dbias[c] = dbias[c] + s;
             }
           }
         });
   }).wait();
}

// k8: flattened 4x8xZ block, tile 32, direct write
void run_k8(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC, int block_size) {
  constexpr int bdx = 4;
  constexpr int bdy = 8;
  int bdz = block_size / 32;
  if (bdz < 1) bdz = 1;
  int L = bdx * bdy * bdz;
  constexpr int Tile = bdy * kVec; // 32
  int Gx = ceil_div_int(OC, Tile);
  int bt_per_block = bdx * bdz;
  int BT = B * T;
  int scratch_elems = bt_per_block * bdy * kVec;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> scratch(sycl::range<1>(scratch_elems), h);
     h.parallel_for<BiasK8>(
         sycl::nd_range<1>(sycl::range<1>(Gx * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           int gx = (int)it.get_group(0);
           int lid = (int)it.get_local_id(0);
           int warp_d = lid % bdx;
           int tmp = lid / bdx;
           int warp_c = tmp % bdy;
           int block_d = tmp / bdy;
           int local_bt = warp_d + bdx * block_d;
           int base = gx * Tile + warp_c * kVec;
           float acc[kVec];
           for (int k = 0; k < kVec; ++k) acc[k] = 0.0f;
           if (base + kVec - 1 < OC || base < OC) {
             for (int idx = local_bt; idx < BT; idx += bt_per_block) {
               for (int k = 0; k < kVec; ++k) {
                 int c = base + k;
                 if (c < OC) acc[k] += dout[idx * OC + c];
               }
             }
           }
           for (int k = 0; k < kVec; ++k)
             scratch[(local_bt * bdy + warp_c) * kVec + k] = acc[k];
           sycl::group_barrier(it.get_group());
           if (local_bt == 0) {
             for (int k = 0; k < kVec; ++k) {
               int c = base + k;
               if (c >= OC) continue;
               float s = 0.0f;
               for (int r = 0; r < bt_per_block; ++r)
                 s += scratch[(r * bdy + warp_c) * kVec + k];
               dbias[c] = dbias[c] + s;
             }
           }
         });
   }).wait();
}

// k9: same tiling as k8 but multi-partition aux + separate row-reduce kernel
void run_k9(sycl::queue &q, float *dbias, const float *dout, int B, int T,
            int OC, int block_size, float *aux) {
  constexpr int bdx = 4;
  constexpr int bdy = 8;
  int bdz = block_size / 32;
  if (bdz < 1) bdz = 1;
  int L = bdx * bdy * bdz;
  constexpr int Tile = bdy * kVec; // 32
  int Gx = ceil_div_int(OC, Tile);
  constexpr int Gy = 4;
  int bt_per_block = bdx * bdz;
  int stride = Gy * bt_per_block;
  int BT = B * T;
  int scratch_elems = bt_per_block * bdy * kVec;
  int G = Gx * Gy;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> scratch(sycl::range<1>(scratch_elems), h);
     h.parallel_for<BiasK9>(
         sycl::nd_range<1>(sycl::range<1>(G * L), sycl::range<1>(L)),
         [=](sycl::nd_item<1> it) {
           int gid = (int)it.get_group(0);
           int lid = (int)it.get_local_id(0);
           int gx = gid % Gx;
           int gy = gid / Gx;
           int warp_d = lid % bdx;
           int tmp = lid / bdx;
           int warp_c = tmp % bdy;
           int block_d = tmp / bdy;
           int local_bt = warp_d + bdx * block_d;
           int base = gx * Tile + warp_c * kVec;
           float acc[kVec];
           for (int k = 0; k < kVec; ++k) acc[k] = 0.0f;
           if (base < OC) {
             for (int idx = gy * bt_per_block + local_bt; idx < BT;
                  idx += stride) {
               for (int k = 0; k < kVec; ++k) {
                 int c = base + k;
                 if (c < OC) acc[k] += dout[idx * OC + c];
               }
             }
           }
           for (int k = 0; k < kVec; ++k)
             scratch[(local_bt * bdy + warp_c) * kVec + k] = acc[k];
           sycl::group_barrier(it.get_group());
           if (local_bt == 0) {
             for (int k = 0; k < kVec; ++k) {
               int c = base + k;
               if (c >= OC) continue;
               float s = 0.0f;
               for (int r = 0; r < bt_per_block; ++r)
                 s += scratch[(r * bdy + warp_c) * kVec + k];
               aux[gy * OC + c] = s;
             }
           }
         });
   }).wait();
  // row-reduce Gy partitions into dbias
  {
    int LR = 256;
    int GR = ceil_div_int(OC, LR) * LR;
    q.submit([&](sycl::handler &h) {
       h.parallel_for<BiasK9Reduce>(sycl::range<1>(GR), [=](sycl::id<1> id) {
         int c = (int)id[0];
         if (c >= OC) return;
         float s = dbias[c];
         for (int g = 0; g < Gy; ++g) s += aux[g * OC + c];
         dbias[c] = s;
       });
     }).wait();
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T OC\n", argv[0]);
    return 2;
  }
  int B = std::atoi(argv[3]);
  int T = std::atoi(argv[4]);
  int OC = std::atoi(argv[5]);
  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  size_t n_dout = (size_t)B * (size_t)T * (size_t)OC;
  float *dout_h = (float *)std::malloc(n_dout * sizeof(float));
  if (!dout_h) return 2;
  if (std::fread(dout_h, sizeof(float), n_dout, fin) != n_dout) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  std::fclose(fin);

  sycl::queue q{sycl::default_selector_v};
  float *d_dout = sycl::malloc_device<float>(n_dout, q);
  float *d_dbias = sycl::malloc_device<float>((size_t)OC, q);
  // aux for k9: Gy(4) * OC
  float *d_aux = sycl::malloc_device<float>((size_t)OC * 4, q);
  if (!d_dout || !d_dbias || !d_aux) return 2;
  q.memcpy(d_dout, dout_h, n_dout * sizeof(float)).wait();

  float *dbias_h = (float *)std::malloc((size_t)OC * sizeof(float));
  if (!dbias_h) return 2;
  std::FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  const int variants[] = {1, 2, 3, 4, 5, 7, 8, 9};
  for (int vi = 0; vi < 8; ++vi) {
    int k = variants[vi];
    q.fill(d_dbias, 0.0f, (size_t)OC).wait();
    if (k == 9) q.fill(d_aux, 0.0f, (size_t)OC * 4).wait();
    switch (k) {
      case 1: run_k1(q, d_dbias, d_dout, B, T, OC, 256); break;
      case 2: run_k2(q, d_dbias, d_dout, B, T, OC); break;
      case 3: run_k3(q, d_dbias, d_dout, B, T, OC, 256); break;
      case 4: run_k4(q, d_dbias, d_dout, B, T, OC, 256); break;
      case 5: run_k5(q, d_dbias, d_dout, B, T, OC, 256); break;
      case 7: run_k7(q, d_dbias, d_dout, B, T, OC); break;
      case 8: run_k8(q, d_dbias, d_dout, B, T, OC, 256); break;
      case 9: run_k9(q, d_dbias, d_dout, B, T, OC, 256, d_aux); break;
      default: return 1;
    }
    q.memcpy(dbias_h, d_dbias, (size_t)OC * sizeof(float)).wait();
    std::fwrite(dbias_h, sizeof(float), (size_t)OC, fout);
  }
  std::fclose(fout);
  sycl::free(d_dout, q);
  sycl::free(d_dbias, q);
  sycl::free(d_aux, q);
  std::free(dout_h);
  std::free(dbias_h);
  return 0;
}

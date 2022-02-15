#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

// pseudo [E, dim]
// mu [K, dim]
// sigma [K, dim]

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define FULLMASK 0xffffffff
#define lane_id threadIdx.x
#define SEG_SHFL_SCAN(v, tmpv, segid, tmps)                                    \
  tmpv = __shfl_down_sync(FULLMASK, v, 1);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 1);                                 \
  if (tmps == segid && lane_id < 31)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 2);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 2);                                 \
  if (tmps == segid && lane_id < 30)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 4);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 4);                                 \
  if (tmps == segid && lane_id < 28)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 8);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 8);                                 \
  if (tmps == segid && lane_id < 24)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 16);                                    \
  tmps = __shfl_down_sync(FULLMASK, segid, 16);                                \
  if (tmps == segid && lane_id < 16)                                           \
    v += tmpv;

__global__ void gaussian(int kernels, int dim, float *pseudo, float *mu,
                         float *inv_sigma, float *gauss) {
  int eid = (blockIdx.x << 5) + threadIdx.x;
  int kid = threadIdx.y;
  float acc = 0;
  for (int d = 0; d < dim; ++d) {
    float tmp = pseudo[eid * dim + d] - mu[kid * dim + d];
    float sig = inv_sigma[kid * dim + d];
    acc += tmp * tmp * sig * sig;
  }
  gauss[eid * kernels + kid] = exp(-0.5 * acc);
}

// Node balance
__global__ void fuseGmm(int kernels, int dim, int embed, int *csrptr,
                        int *colind, float *node_feat, float *pseudo, float *mu,
                        float *inv_sigma, float *out) {
  extern __shared__ float shmem[];
  int ssize = blockDim.x;
  int rid = blockIdx.x;
  int kid = threadIdx.z;
  int fid = (threadIdx.y << 5) + threadIdx.x;
  int lb = csrptr[rid];
  int hb = csrptr[rid + 1];
  int ptr = lb;
  float acc = 0;
  // out[rid * embed + fid] = 0;
  for (; ptr < hb; ptr += ssize) {
    float gauss = 0;
    for (int d = 0; d < dim; ++d) {
      float tmp = pseudo[(ptr + threadIdx.x) * dim + d] - mu[kid * dim + d];
      float sig = inv_sigma[kid * dim + d];
      gauss += tmp * tmp * sig * sig;
    }
    gauss = exp(-0.5 * gauss);
    shmem[kid * kernels + threadIdx.x] = gauss;
    __syncwarp();

    for (int cnt = 0; cnt < ssize && cnt + ptr < hb; cnt++) {
      int col = colind[cnt + ptr];
      acc += node_feat[col * embed * kernels + kid * embed + fid] *
             shmem[kid * kernels + cnt];
    }
  }
  if (fid < embed)
    out[rid * embed * kernels + kid * embed + fid] = acc;
}

__global__ void fuseGmmEdgeBalance(int nodes, int nnz, int kernels, int dim,
                                   int embed, int *csrptr, int *colind,
                                   float *node_feat, float *pseudo, float *mu,
                                   float *inv_sigma, float *out) {
  int eid = (blockIdx.x << 5) + threadIdx.x;
  int kid = threadIdx.z;
  int fid = (threadIdx.y << 5) + threadIdx.x;

  // out[rid * embed + fid] = 0;
  float gauss = 0;
  for (int d = 0; d < dim; ++d) {
    float tmp = pseudo[eid * dim + d] - mu[kid * dim + d];
    float sig = inv_sigma[kid * dim + d];
    gauss += tmp * tmp * sig * sig;
  }
  gauss = exp(-0.5 * gauss);
  int col = 0;
  float acc = 0;
  if (eid < nnz) {
    col = colind[eid];
    acc = node_feat[col * embed * kernels + kid * embed + fid] * gauss;
  }
  // is_start
  int rid = findRow(csrptr, eid, 0, nodes);
  int r_left = rid;
  r_left = __shfl_up_sync(FULLMASK, r_left, 1);
  bool is_start = (r_left != rid) || (threadIdx.x == 0);
  // shfl
  int tmpv = 0;
  int tmps = 0;
  SEG_SHFL_SCAN(acc, tmpv, rid, tmps);
  if (is_start && rid != -1) {
    if (fid < embed)
      atomicAdd(out + rid * embed * kernels + kid * embed + fid, acc);
  }
}

__global__ void gmm_stash(int kernels, int dim, int embed, int *csrptr,
                          int *colind, float *node_feat, float *pseudo,
                          float *mu, float *inv_sigma, float *out,
                          float *gaussian) {
  extern __shared__ float shmem[];
  int ssize = blockDim.x;
  int rid = blockIdx.x;
  int kid = threadIdx.z;
  int fid = (threadIdx.y << 5) + threadIdx.x;
  int lb = csrptr[rid];
  int hb = csrptr[rid + 1];
  int ptr = lb;
  float acc = 0;
  // out[rid * embed + fid] = 0;
  for (; ptr < hb; ptr += ssize) {
    float gauss = 0;
    for (int d = 0; d < dim; ++d) {
      float tmp = pseudo[(ptr + threadIdx.x) * dim + d] - mu[kid * dim + d];
      float sig = inv_sigma[kid * dim + d];
      gauss += tmp * tmp * sig * sig;
    }
    gauss = exp(-0.5 * gauss);
    gaussian[(ptr + threadIdx.x) * kernels + kid] = gauss;
    shmem[kid * kernels + threadIdx.x] = gauss;
    __syncwarp();
    for (int cnt = 0; cnt < ssize && cnt + ptr < hb; cnt++) {
      int col = colind[cnt + ptr];
      acc += node_feat[col * embed * kernels + kid * embed + fid] *
             shmem[kid * kernels + cnt];
    }
  }
  if (fid < embed)
    out[rid * embed * kernels + kid * embed + fid] = acc;
}

__global__ void gaussian_bp(int edges, int kernels, int dim, float *pseudo,
                            float *mu, float *inv_sigma, float *grad_gauss,
                            float *pseudo_out, float *sigma_out,
                            float *mu_out) {
  int eid = (blockIdx.x << 5) + threadIdx.x;
  int kid = threadIdx.y;
  float tmp_mout = 0, tmp_pout = 0, tmp_sout = 0, tmp_gauss = 0;
  for (int d = 0; d < dim; ++d) {
    float sig = inv_sigma[kid * dim + d];
    float pse = pseudo[eid * dim + d];
    float m = mu[kid * dim + d];
    float tmp = (pse - m) * sig;
    tmp_mout += sig * tmp;
    tmp_pout += -tmp_mout;
    tmp_sout += tmp * (m - pse);
    tmp_gauss += tmp * tmp;
  }
  tmp_gauss = exp(-0.5 * tmp_gauss) * grad_gauss[eid * kernels + kid];
  tmp_mout *= tmp_gauss;
  tmp_sout *= tmp_gauss;
  AllReduce<float>(tmp_mout, 16, 32);
  AllReduce<float>(tmp_sout, 16, 32);
  for (int d = 0; d < dim; ++d)
    atomicAdd(&pseudo_out[eid * dim + d], tmp_pout * tmp_gauss);
  if (threadIdx.x == 0) {
    for (int d = 0; d < dim; ++d) {
      atomicAdd(&sigma_out[kid * dim + d], tmp_sout);
      atomicAdd(&mu_out[kid * dim + d], tmp_mout);
    }
  }
}

torch::Tensor gaussian_cuda(torch::Tensor pseudo, torch::Tensor mu,
                            torch::Tensor inv_sigma) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto gauss = torch::empty({edges, K}, options);
  gaussian<<<dim3(CEIL(edges, 32), 1, 1), dim3(32, K, 1)>>>(
      K, dim, pseudo.data_ptr<float>(), mu.data_ptr<float>(),
      inv_sigma.data_ptr<float>(), gauss.data_ptr<float>());
  return gauss;
}

torch::Tensor gmmconv_cuda(torch::Tensor csrptr, torch::Tensor colind,
                           torch::Tensor node_feat, torch::Tensor pseudo,
                           torch::Tensor mu, torch::Tensor inv_sigma) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  const auto nodes = csrptr.size(0) - 1;
  const auto embed = node_feat.size(2);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nodes, K, embed}, options);
  fuseGmm<<<dim3(nodes, 1, 1), dim3(32, CEIL(embed, 32), K),
            32 * K * sizeof(float)>>>(
      K, dim, embed, csrptr.data_ptr<int>(), colind.data_ptr<int>(),
      node_feat.data_ptr<float>(), pseudo.data_ptr<float>(),
      mu.data_ptr<float>(), inv_sigma.data_ptr<float>(), out.data_ptr<float>());
  return out;
}

torch::Tensor gmmconveb_cuda(torch::Tensor csrptr, torch::Tensor colind,
                             torch::Tensor node_feat, torch::Tensor pseudo,
                             torch::Tensor mu, torch::Tensor inv_sigma) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  const auto nodes = csrptr.size(0) - 1;
  const auto embed = node_feat.size(2);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nodes, K, embed}, options);
  fuseGmmEdgeBalance<<<dim3(CEIL(edges, 32), 1, 1),
                       dim3(32, CEIL(embed, 32), K)>>>(
      nodes, edges, K, dim, embed, csrptr.data_ptr<int>(),
      colind.data_ptr<int>(), node_feat.data_ptr<float>(),
      pseudo.data_ptr<float>(), mu.data_ptr<float>(),
      inv_sigma.data_ptr<float>(), out.data_ptr<float>());
  return out;
}

std::vector<torch::Tensor>
gmmconv_stash_cuda(torch::Tensor csrptr, torch::Tensor colind,
                   torch::Tensor node_feat, torch::Tensor pseudo,
                   torch::Tensor mu, torch::Tensor inv_sigma) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  const auto nodes = csrptr.size(0) - 1;
  const auto embed = node_feat.size(2);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nodes, K, embed}, options);
  auto gaussian = torch::empty({edges, K}, options);
  gmm_stash<<<dim3(nodes, 1, 1), dim3(32, CEIL(embed, 32), K),
              32 * K * sizeof(float)>>>(
      K, dim, embed, csrptr.data_ptr<int>(), colind.data_ptr<int>(),
      node_feat.data_ptr<float>(), pseudo.data_ptr<float>(),
      mu.data_ptr<float>(), inv_sigma.data_ptr<float>(), out.data_ptr<float>(),
      gaussian.data_ptr<float>());
  return {out, gaussian};
}

std::vector<torch::Tensor> gaussian_bp_cuda(torch::Tensor pseudo,
                                            torch::Tensor mu,
                                            torch::Tensor inv_sigma,
                                            torch::Tensor grad_out) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto pseudo_out = torch::empty({edges, dim}, options);
  auto sigma_out = torch::empty({K, dim}, options);
  auto mu_out = torch::empty({K, dim}, options);

  gaussian_bp<<<dim3(CEIL(edges, 32), 1, 1), dim3(32, K, 1)>>>(
      edges, K, dim, pseudo.data_ptr<float>(), mu.data_ptr<float>(),
      inv_sigma.data_ptr<float>(), grad_out.data_ptr<float>(),
      pseudo_out.data_ptr<float>(), sigma_out.data_ptr<float>(),
      mu_out.data_ptr<float>());
  return {pseudo_out, mu_out, sigma_out};
}

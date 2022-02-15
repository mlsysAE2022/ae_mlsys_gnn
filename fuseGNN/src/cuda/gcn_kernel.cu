#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>


/*
 * get degree
 */

#define THREADS 256
#define BLOCKS(N, T) (N + T - 1)/T

// when using CSR/CSC format
template <typename scalar_t>
__global__ void from_ptr(
    int* __restrict__ tar_ptr,
    scalar_t* __restrict__ degree,
    int num_nodes
){
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_nodes; tid += stride){
        degree[tid] = tar_ptr[tid + 1] - tar_ptr[tid] + 1;
    }
}


template <typename scalar_t, unsigned int blockSize>
__device__ void smem_reduction(volatile scalar_t* sdata, unsigned int tid){
    if (blockSize >= 1024){
        if (tid < 512){
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512){
        if (tid < 256){
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256){
        if (tid < 128){
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128){
        if (tid < 64){
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32){
        if (blockSize >= 64)sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32)sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16)sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)sdata[tid] += sdata[tid + 1];
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void from_weight(
    scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ degree,
    int* __restrict__ tar_ptr
){
    __shared__ scalar_t deg[blockSize];
    unsigned int tid = threadIdx.x;
    deg[tid] = 0;
    unsigned int tar_id = blockIdx.x;
    for (unsigned int e_idx = tar_ptr[tar_id] + threadIdx.x; e_idx < tar_ptr[tar_id + 1]; e_idx += blockDim.x){
        deg[tid] += edge_weight[e_idx];
    }
    __syncthreads();
    smem_reduction<scalar_t, blockSize>(deg, tid);
    if (tid == 0) degree[tar_id] = deg[0] + 1;
}

// When using COO format
template <typename scalar_t>
__global__ void scatter_add(
    scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ degree,
    int* __restrict__ tar_index,
    unsigned int num_edge
){
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_edge; tid += stride){
        atomicAdd(&degree[tar_index[tid]], edge_weight[tid]);
    }
}

torch::Tensor get_degree_cuda(
    torch::Tensor tar_ptr,
    torch::Tensor src_index,
    torch::optional<torch::Tensor> optional_edge_weight,
    int num_nodes,
    bool tar_to_src
){
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tar_ptr.device());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor degree;

    if (optional_edge_weight.has_value()){
        degree = torch::empty({num_nodes,}, options);
        torch::Tensor edge_weight;
        edge_weight = optional_edge_weight.value().contiguous();
        if (tar_to_src){
            AT_DISPATCH_FLOATING_TYPES(degree.type(), "get degree from weight", ([&]{
                from_weight<scalar_t, 64><<<num_nodes, 64, 0, stream>>>(
                    edge_weight.data<scalar_t>(), degree.data<scalar_t>(),
                    tar_ptr.data<int>());
            }));
        }else{
            degree = torch::ones({num_nodes,}, options);

            AT_DISPATCH_FLOATING_TYPES(degree.type(), "get degree from weight scatter", ([&]{
                scatter_add<scalar_t><<<BLOCKS(edge_weight.size(0), THREADS), THREADS, 0, stream>>>(
                    edge_weight.data<scalar_t>(), degree.data<scalar_t>(),
                    src_index.data<int>(), edge_weight.size(0));
            }));
        }
    }else{
        degree = torch::empty({num_nodes,}, options);
        AT_DISPATCH_FLOATING_TYPES(degree.type(), "get degree from ptr", ([&]{
            from_ptr<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
                tar_ptr.data<int>(), degree.data<scalar_t>(), num_nodes
            );
        }));
    }

    return degree;
}


template <typename scalar_t>
__global__ void update_weight(
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ out_edge_weight,
    const scalar_t* __restrict__ degree,
    int num_edge
){
    for(unsigned int tid=blockIdx.x * blockDim.x + threadIdx.x; tid < num_edge; tid += gridDim.x * blockDim.x){
        scalar_t res = edge_weight[tid] / sqrtf(degree[src_index[tid]])/sqrtf(degree[tar_index[tid]]);
        if (isinf(res)) res = 0;
        out_edge_weight[tid] = res;
    }
}


template <typename scalar_t>
__global__ void get_weight(
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ degree,
    int num_edge
){
    for(unsigned int tid=blockIdx.x * blockDim.x + threadIdx.x; tid < num_edge; tid += gridDim.x * blockDim.x){
        scalar_t res = 1/sqrtf(degree[src_index[tid]] * degree[tar_index[tid]]);
        if (isinf(res)) res = 0;
        edge_weight[tid] = res;
    }
}


// CUDA Edge processing declaration
std::vector<torch::Tensor> gcn_gar_egde_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight,
    bool tar_to_src
){
    // Step 1: get degree
    auto degree = get_degree_cuda(tar_ptr, src_index, optional_edge_weight, num_nodes, tar_to_src);

    // Step 3: initialize the edge_weight with 1s if not provided
    unsigned int Ne = src_index.size(0);
    torch::Tensor edge_weight;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (optional_edge_weight.has_value()){
        edge_weight = optional_edge_weight.value().contiguous();
        auto out_edge_weight = torch::empty_like(edge_weight);
        AT_DISPATCH_FLOATING_TYPES(edge_weight.type(), "update weight", ([&]{
            update_weight<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
                src_index.data<int>(), tar_index.data<int>(),
                edge_weight.data<scalar_t>(),
                out_edge_weight.data<scalar_t>(),
                degree.data<scalar_t>(), src_index.size(0)
            );
        }));
        return {out_edge_weight, degree};
    }else{   
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(src_index.device());
        edge_weight = torch::empty({Ne,}, options);
        AT_DISPATCH_FLOATING_TYPES(edge_weight.type(), "update weight", ([&]{
            get_weight<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
                src_index.data<int>(), tar_index.data<int>(),
                edge_weight.data<scalar_t>(),
                degree.data<scalar_t>(), src_index.size(0)
            );
        }));
        return {edge_weight, degree};
    }
}


// CUDA Edge processing declaration
std::vector<torch::Tensor> gcn_gas_edge_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight,
    bool tar_to_src
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unsigned int Ne = src_index.size(0);
    // Step 0: get edge_weight
    torch::Tensor edge_weight;

    if (optional_edge_weight.has_value()){
        edge_weight = optional_edge_weight.value().contiguous();
    }else{
        auto options_w = torch::TensorOptions().dtype(torch::kFloat32).device(src_index.device());
        edge_weight = torch::ones({Ne,}, options_w);
    }
    // Step 1: get degree
    auto options_d = torch::TensorOptions().dtype(torch::kFloat32).device(tar_index.device());

    auto degree = torch::ones({num_nodes,}, options_d);

    if (tar_to_src){
        AT_DISPATCH_FLOATING_TYPES(degree.type(), "get degree from weight scatter", ([&]{
            scatter_add<scalar_t><<<BLOCKS(edge_weight.size(0), THREADS), THREADS, 0, stream>>>(
                edge_weight.data<scalar_t>(), degree.data<scalar_t>(),
                tar_index.data<int>(), edge_weight.size(0));
        }));
    }else{
        AT_DISPATCH_FLOATING_TYPES(degree.type(), "get degree from weight scatter", ([&]{
            scatter_add<scalar_t><<<BLOCKS(edge_weight.size(0), THREADS), THREADS, 0, stream>>>(
                edge_weight.data<scalar_t>(), degree.data<scalar_t>(),
                src_index.data<int>(), edge_weight.size(0));
        }));
    }
    auto out_edge_weight = torch::empty_like(edge_weight);

    AT_DISPATCH_FLOATING_TYPES(edge_weight.type(), "update weight", ([&]{
        update_weight<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
            src_index.data<int>(), tar_index.data<int>(),
            edge_weight.data<scalar_t>(),
            out_edge_weight.data<scalar_t>(),
            degree.data<scalar_t>(), src_index.size(0)
        );
    }));

    return {out_edge_weight, degree};
}

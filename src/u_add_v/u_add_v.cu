#include <cuda.h>
#include <torch/types.h>

__global__ void u_add_v_forward_kernel(
    const int f,
    const int *row_ptr, const int *col_ind,
    const float *row_val,
    const float *col_val,
    float *result_csr)
{
    int rid = blockIdx.x;
    int fid = blockIdx.y;
    int lb = row_ptr[rid];
    int hb = row_ptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int loop = (hb - lb + 31) / 32;
    float val_row = row_val[rid * f + fid];

    for (int j = 0; j < loop; j++)
    {
        int pid = ptr + (j << 5);
        if (pid < hb)
        {
            int cid = col_ind[pid];
            result_csr[pid * f + fid] = val_row + col_val[cid * f + fid];
        }
    }
}

torch::Tensor u_add_v_forward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor row_val, torch::Tensor col_val)
{
    const auto f = row_val.size(1);
    const auto nnz = col_ind.size(0);
    const auto m = row_ptr.size(0) - 1;
    auto devid = row_val.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out = torch::empty({nnz, f}, options);
    u_add_v_forward_kernel<<<dim3(m, f, 1), dim3(32, 1, 1)>>>(f, row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(), row_val.data_ptr<float>(), col_val.data_ptr<float>(), out.data_ptr<float>());
    // printf("forwarding\n");
    return out;
}

__global__ void u_add_v_backward_kernel(
    const int f,
    const int *row_ptr, const int *col_ind,
    const float *grad_csr,
    float *grad_row,
    float *grad_col)
{
    int rid = blockIdx.x;
    int fid = blockIdx.y;
    int lb = row_ptr[rid];
    int hb = row_ptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int loop = (hb - lb + 31) / 32;

    float grad_row_sum = 0;
    for (int j = 0; j < loop; j++)
    {
        int pid = ptr + (j << 5);
        float grad_out = 0;
        int eid = pid * f + fid;
        if (pid < hb)
        {
            int cid = col_ind[pid];
            grad_out = grad_csr[eid];
            atomicAdd(&grad_col[cid * f + fid], grad_out);
        }
        __syncwarp();
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            grad_out += __shfl_xor_sync(0xffffffff, grad_out, stride, 32);
        }
        grad_row_sum += grad_out;
    }
    if (threadIdx.x == 0)
        grad_row[rid * f + fid] = grad_row_sum;
}

std::vector<torch::Tensor> u_add_v_backward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor grad_csr)
{
    const auto f = grad_csr.size(1);
    const auto m = row_ptr.size(0) - 1;
    auto devid = row_ptr.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto grad_row = torch::empty({m, f}, options);
    auto grad_col = torch::empty({m, f}, options);
    u_add_v_backward_kernel<<<dim3(m, f, 1), dim3(32, 1, 1)>>>(f, row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(), grad_csr.data_ptr<float>(), grad_row.data_ptr<float>(), grad_col.data_ptr<float>());
    return {grad_row, grad_col};
}
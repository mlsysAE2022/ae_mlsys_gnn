#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

void assertTensor(torch::Tensor &T, torch::ScalarType type)
{
    assert(T.is_contiguous());
    assert(T.device().type() == torch::kCUDA);
    assert(T.dtype() == type);
}

torch::Tensor u_add_v_forward_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor row_val,
    torch::Tensor col_val);

torch::Tensor u_add_v_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor row_val,
    torch::Tensor col_val)
{
    // printf("u_add_v forwarding\n");
    assert(row_ptr.device().type() == torch::kCUDA);
    assert(col_ind.device().type() == torch::kCUDA);
    assert(row_val.device().type() == torch::kCUDA);
    assert(col_val.device().type() == torch::kCUDA);
    assert(row_ptr.is_contiguous());
    assert(col_ind.is_contiguous());
    assert(row_val.is_contiguous());
    assert(col_val.is_contiguous());
    assertTensor(row_ptr, torch::kInt32);
    assertTensor(col_ind, torch::kInt32);
    assertTensor(row_val, torch::kFloat32);
    assertTensor(col_val, torch::kFloat32);
    return u_add_v_forward_cuda(row_ptr, col_ind, row_val, col_val);
}

std::vector<torch::Tensor> u_add_v_backward_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor grad_csr);

std::vector<torch::Tensor> u_add_v_backward(
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor grad_csr)
{
    assertTensor(row_ptr, torch::kInt32);
    assertTensor(col_ind, torch::kInt32);
    assertTensor(grad_csr, torch::kFloat32);
    return u_add_v_backward_cuda(row_ptr, col_ind, grad_csr);
}

PYBIND11_MODULE(u_add_v, m)
{
    m.doc() = "u_add_v kernel. ";
    m.def("u_add_v_forward", &u_add_v_forward, "u_add_v_forward op");
    m.def("u_add_v_backward", &u_add_v_backward, "u_add_v_backward op");
}

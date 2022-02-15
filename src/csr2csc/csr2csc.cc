#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal);

std::vector<torch::Tensor> csr2csc(torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor csr_data) {
  assert(rowptr.device().type() == torch::kCUDA);
  assert(colind.device().type() == torch::kCUDA);
  assert(csr_data.device().type() == torch::kCUDA);
  assert(rowptr.is_contiguous());
  assert(colind.is_contiguous());
  assert(csr_data.is_contiguous());
  assert(rowptr.dtype() == torch::kInt32);
  assert(colind.dtype() == torch::kInt32);
  assert(csr_data.dtype() == torch::kFloat32);
  return csr2csc_cuda(rowptr, colind, csr_data);
}

PYBIND11_MODULE(csrtocsc, m) {
  m.doc() = "csr2csc provides the format transformation";
  m.def("csr2csc", &csr2csc, "csr2csc");
}
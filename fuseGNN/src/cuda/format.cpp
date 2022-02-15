#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> csr2csc_cuda(
    torch::Tensor inPtr,
    torch::Tensor inInd,
    torch::Tensor inVal,
    int num_row
);


std::vector<torch::Tensor> csr2csc(
    torch::Tensor inPtr,
    torch::Tensor inInd,
    torch::Tensor inVal,
    int num_row
){
    return csr2csc_cuda(inPtr, inInd, inVal, num_row);
}

torch::Tensor coo2csr_cuda(
    torch::Tensor cooRowInd,
    int num_row
);

torch::Tensor coo2csr(
    torch::Tensor cooRowInd,
    int num_row
){
    return coo2csr_cuda(cooRowInd, num_row);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("csr2csc", &csr2csc, "Converter between CSC and CSR");
    m.def("coo2csr", &coo2csr, "Convert COO to CSR");
}
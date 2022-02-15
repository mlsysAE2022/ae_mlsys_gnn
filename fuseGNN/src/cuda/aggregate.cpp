#include <torch/extension.h>
#include <vector>


torch::Tensor fused_gar_f_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight
);


torch::Tensor fused_gar_f(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight
){
    return fused_gar_f_cuda(feature, src_index, tar_ptr, edge_weight, self_edge_weight);
}


std::vector<torch::Tensor> fused_gar_b_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor tar_index,
    torch::Tensor src_ptr,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight,
    bool require_edge_weight
);


std::vector<torch::Tensor> fused_gar_b(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor tar_index,
    torch::Tensor src_ptr,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight,
    bool require_edge_weight
){
    return fused_gar_b_cuda(grad_out, feature, tar_index, src_ptr, edge_weight, self_edge_weight, require_edge_weight);
}


torch::Tensor fused_gas_f_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight
);


torch::Tensor fused_gas_f(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight
){
    return fused_gas_f_cuda(feature, src_index, tar_index, edge_weight, self_edge_weight);
}


std::vector<torch::Tensor> fused_gas_b_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight,
    bool require_edge_weight
);


std::vector<torch::Tensor> fused_gas_b(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight,
    torch::Tensor self_edge_weight,
    bool require_edge_weight
){
    return fused_gas_b_cuda(grad_out, feature, src_index, tar_index, edge_weight, self_edge_weight, require_edge_weight);
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("fused_gar_f", &fused_gar_f, "fused AGG GAR forward");
    m.def("fused_gar_b", &fused_gar_b, "fused AGG GAR backward");
    m.def("fused_gas_f", &fused_gas_f, "fused AGG GAS forward");
    m.def("fused_gas_b", &fused_gas_b, "fused AGG GAS backward");
}

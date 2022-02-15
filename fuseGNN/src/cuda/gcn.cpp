#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> gcn_gar_egde_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight,
    bool tar_to_src
);

std::vector<torch::Tensor> gcn_gar_egde_weight(
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight,
    bool tar_to_src
){
    return gcn_gar_egde_weight_cuda(src_index, tar_ptr, tar_index, num_nodes, optional_edge_weight, tar_to_src);
}

std::vector<torch::Tensor> gcn_gas_edge_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight,
    bool tar_to_src
);


std::vector<torch::Tensor> gcn_gas_edge_weight(
    torch::Tensor src_index,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight,
    bool tar_to_src
){
    return gcn_gas_edge_weight_cuda(src_index, tar_index, num_nodes, optional_edge_weight, tar_to_src);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("gcn_gar_edge_weight", &gcn_gar_egde_weight, "gcn_gar_edge_weight");
    m.def("gcn_gas_edge_weight", &gcn_gas_edge_weight, "gcn_gas_edge_weight");
}
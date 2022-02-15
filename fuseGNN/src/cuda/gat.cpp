#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> gat_gar_edge_weight_cuda(
    torch::Tensor e_pre,
    torch::Tensor src_ptr,
    torch::Tensor tar_index,
    float negative_slope
);


std::vector<torch::Tensor> gat_gar_edge_weight(
    torch::Tensor e_pre,
    torch::Tensor src_ptr,
    torch::Tensor tar_index,
    float negative_slope
){
    return gat_gar_edge_weight_cuda(e_pre, src_ptr, tar_index, negative_slope);
}


std::vector<torch::Tensor> gat_gas_edge_weight_cuda(
    torch::Tensor e_pre,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    float negative_slope
);


std::vector<torch::Tensor> gat_gas_edge_weight(
    torch::Tensor e_pre,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    float negative_slope
){
    return gat_gas_edge_weight_cuda(e_pre, src_index, tar_index, negative_slope);
}


std::vector<torch::Tensor> gat_gar_edge_weight_b_cuda(
    torch::Tensor grad_alpha_self,
    torch::Tensor grad_alpha,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor mask_lrelu,
    torch::Tensor mask_lrelu_self,
    torch::Tensor e,
    torch::Tensor e_self,
    torch::Tensor e_sum,
    torch::Tensor alpha_self,
    torch::Tensor alpha
);

std::vector<torch::Tensor> gat_gar_edge_weight_b(
    torch::Tensor grad_alpha_self,
    torch::Tensor grad_alpha,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor mask_lrelu,
    torch::Tensor mask_lrelu_self,
    torch::Tensor e,
    torch::Tensor e_self,
    torch::Tensor e_sum,
    torch::Tensor alpha_self,
    torch::Tensor alpha
){
    return gat_gar_edge_weight_b_cuda(grad_alpha_self, grad_alpha, src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("gat_gar_edge_weight", &gat_gar_edge_weight, "gat_gar_edge_weight");
    m.def("gat_gas_edge_weight", &gat_gas_edge_weight, "gat_gas_edge_weight");
    m.def("gat_gar_edge_weight_b", &gat_gar_edge_weight_b, "gat_gar_edge_weight_b");
}

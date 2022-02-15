from torch.utils.cpp_extension import load
import torch
import os

# path = os.path.join(os.path.dirname(__file__))
path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


fused_edgeconv=load(
    name="fused_edgeconv",
    sources=[os.path.join(path, "src/fused_edgeconv/fused_edgeconv.cpp"), os.path.join(path, "src/fused_edgeconv/fused_edgeconv.cu")],
    verbose=True,
)

def fused_edgeconv_op(k,src_ind,h_src,h_dst):
    return FusedEdgeConvFunction.apply(k,src_ind,h_src,h_dst)

class FusedEdgeConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,src_ind,h_src,h_dst):
        out_feat,max_idx=fused_edgeconv.edgeconv_forward(k,src_ind,h_src,h_dst)
        ctx.save_for_backward(max_idx)
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        max_idx=ctx.saved_tensors[0].int()       
        grad_src=fused_edgeconv.edgeconv_backward(grad_out,max_idx)
        return None,None,grad_src,grad_out

from torch.utils.cpp_extension import load
import torch
import os

# path = os.path.join(os.path.dirname(__file__))
path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
path=os.path.join(path,'src')

fused_gat = load(
    name="fused_gat",
    sources=[os.path.join(path, "fused_gat/fused_gat.cpp"), os.path.join(path, "fused_gat/fused_gat.cu")],
    verbose=True,
)

fused_gat_SM = load(
    name="fused_gat_SM",
    sources=[os.path.join(path, "fused_gat/fused_gat_SM.cpp"), os.path.join(path, "fused_gat/fused_gat_SM.cu")],
    verbose=False,
)

fused_gat_stash = load(
    name="fused_gat_stash",
    sources=[os.path.join(path, "fused_gat/fused_gat_stash.cpp"), os.path.join(path, "fused_gat/fused_gat_stash.cu")],
    verbose=False,
)

def fused_gat_fusescatter(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,permute):
    return FusedGATFunction.apply(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,permute)

def fused_gat_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,save_memory=True):
    return FusedGATFunction_SM.apply(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat)

def fused_gat_stash_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,permute):
    return FusedGATFunction_stash.apply(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,permute)

class FusedGATFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,permute):
        out_feat,edge_relu_csr,edge_softmax_csr=fused_gat.gat_forward(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat)
        ctx.save_for_backward(row_ptr,col_ind,col_ptr,row_ind,permute,edge_softmax_csr,edge_relu_csr,in_feat)
        ctx.negative_slope=negative_slope      
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        row_ptr,col_ind,col_ptr,row_ind,permute,edge_softmax_csr,edge_relu_csr,in_feat=ctx.saved_tensors
        grad_out=grad_out.contiguous()
        # print('start backward')
        grad_feat,grad_attn_row,grad_attn_col=fused_gat.gat_backward(ctx.negative_slope,row_ptr,col_ind,col_ptr,row_ind,permute,edge_softmax_csr,edge_relu_csr,in_feat,grad_out)
        return grad_attn_row,grad_attn_col,None,None,None,None,None,grad_feat,None

class FusedGATFunction_SM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat):
        out_feat,edge_max,edge_sum=fused_gat_SM.gat_forward(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat)
        ctx.save_for_backward(row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col)
        ctx.negative_slope=negative_slope      
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col=ctx.saved_tensors
        grad_out=grad_out.contiguous()
        # print('start backward')
        grad_feat,grad_attn_row,grad_attn_col=fused_gat_SM.gat_backward(
            ctx.negative_slope,row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col,grad_out)
        # print('end backward')
        # print(torch.isnan(grad_feat).sum())
        # print(torch.isnan(grad_attn_row).sum())
        # print(torch.isnan(grad_attn_col).sum())
        return grad_attn_row,grad_attn_col,None,None,None,None,None,grad_feat,None

class FusedGATFunction_stash(torch.autograd.Function):
    @staticmethod
    def forward(ctx,attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,permute):
        out_feat,edge_relu_csr,edge_softmax_csr=fused_gat_stash.gat_forward(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat)
        ctx.save_for_backward(row_ptr,col_ind,col_ptr,row_ind,permute,edge_softmax_csr,edge_relu_csr,in_feat,attn_row,attn_col)
        ctx.negative_slope=negative_slope      
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        row_ptr,col_ind,col_ptr,row_ind,permute,edge_softmax_csr,edge_relu_csr,in_feat,attn_row,attn_col=ctx.saved_tensors
        grad_out=grad_out.contiguous()
        # print('start backward')
        grad_feat,grad_attn_row,grad_attn_col=fused_gat_stash.gat_backward(ctx.negative_slope,row_ptr,col_ind,col_ptr,row_ind,permute,edge_relu_csr,edge_softmax_csr,in_feat,attn_row,attn_col,grad_out)
        return grad_attn_row,grad_attn_col,None,None,None,None,None,grad_feat,None
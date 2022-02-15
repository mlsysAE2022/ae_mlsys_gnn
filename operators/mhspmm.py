import os

import torch
from torch.utils.cpp_extension import load

path = os.path.join(os.path.dirname(os.path.dirname(__file__)))

mhspmm = load(
    name="mhspmm",
    sources=[os.path.join(path, "src/spmm/mhspmm.cc"), os.path.join(path, "src/spmm/mhspmm.cu")],
    verbose=False,
)
mhsddmm = load(
    name="mhsddmm",
    sources=[os.path.join(path, "src/sddmm/mhsddmm.cc"), os.path.join(path, "src/sddmm/mhsddmm.cu")],
    verbose=False,
)
mhtranspose = load(
    name="mhtranspose",
    sources=[os.path.join(path, "src/csr2csc/mhtranspose.cc"), os.path.join(path, "src/csr2csc/mhtranspose.cu")],
    verbose=False,
)

def csrmhspmm(rowptr, colind, colptr, rowind, permute, feat, attention):
    return MHSPMMFunction.apply(rowptr, colind, colptr, rowind, permute, feat, attention)

class MHSPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, permute, feat, attention):
        out = mhspmm.mhspmm(rowptr, colind, attention, feat)
        ctx.save_for_backward(rowptr, colind, colptr, rowind, permute, feat, attention)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rowptr, colind, colptr, rowind, permute, feat, attention = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        edge_feat = mhtranspose.mhtranspose(permute, attention)
        grad_feat = mhspmm.mhspmm(colptr, rowind, edge_feat, grad_out)
        grad_edge_weight = mhsddmm.mhsddmm(rowptr, colind, grad_out, feat)
        return None, None, None, None, None, grad_feat, grad_edge_weight

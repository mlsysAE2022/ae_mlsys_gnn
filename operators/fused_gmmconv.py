import os
import torch
from torch.utils.cpp_extension import load
import time

path = os.path.join(os.path.dirname(os.path.dirname(__file__)))

# SPMM
# try:
gmm = load(
    name="gmmconv",
    sources=[
        os.path.join(path, "src/fused_gmmconv/gmmconv.cc"),
        os.path.join(path, "src/fused_gmmconv/gmmconv.cu"),
    ],
    verbose=True,
)
mhsddmm = load(
    name="mhsddmm",
    sources=[
        os.path.join(path, "src/sddmm/mhsddmm.cc"),
        os.path.join(path, "src/sddmm/mhsddmm.cu"),
    ],
    verbose=True,
)
mhspmm = load(
    name="mhspmm",
    sources=[
        os.path.join(path, "src/spmm/mhspmm.cc"),
        os.path.join(path, "src/spmm/mhspmm.cu"),
    ],
    verbose=True,
)
mhtranspose = load(
    name="mhtranspose",
    sources=[os.path.join(path, "src/csr2csc/mhtranspose.cc"), os.path.join(path, "src/csr2csc/mhtranspose.cu")],
    verbose=False,
)

def GmmConvFuse(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
    return GmmFuseFunction.apply(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma)

class GmmFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
        out = gmm.gmmconv(rowptr, colind, node_feat, pseudo, mu, sigma) # (E, K, F)
        ctx.save_for_backward(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma)
        return out

    @staticmethod
    def backward(ctx, grad_out): # (E, F)
        rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        guassian_grad = mhsddmm.mhsddmm(rowptr, colind, grad_out, node_feat)
        pseudo_tran = mhtranspose.mhtranspose(permute, pseudo) #
        node_feat_grad = gmm.gmmconv(colptr, rowind, grad_out, pseudo_tran, mu, sigma)
        pseudo_out, mu_out, sigma_out = gmm.GaussianBp(pseudo, mu, sigma, guassian_grad)
        return None, None, None, None, None, node_feat_grad, pseudo_out, mu_out, sigma_out


def GmmConvFuseEb(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
    return GmmFuseEbFunction.apply(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma)

class GmmFuseEbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
        out = gmm.gmmconveb(rowptr, colind, node_feat, pseudo, mu, sigma) # (E, K, F)
        ctx.save_for_backward(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma)
        return out

    @staticmethod
    def backward(ctx, grad_out): # (E, F)
        rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        guassian_grad = mhsddmm.mhsddmm(rowptr, colind, grad_out, node_feat)
        pseudo_tran = mhtranspose.mhtranspose(permute, pseudo) #
        node_feat_grad = gmm.gmmconveb(colptr, rowind, grad_out, pseudo_tran, mu, sigma)
        pseudo_out, mu_out, sigma_out = gmm.GaussianBp(pseudo, mu, sigma, guassian_grad)
        return None, None, None, None, None, node_feat_grad, pseudo_out, mu_out, sigma_out


def GaussianCp(pseudo, mu, sigma):
    return GaussianFunction.apply(pseudo, mu, sigma)

class GaussianFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pseudo, mu, sigma):
        out = gmm.Gaussian(pseudo, mu, sigma)
        ctx.save_for_backward(pseudo, mu, sigma)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # print('edge softmax backward')
        pseudo, mu, sigma = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        pseudo_out, mu_out, sigma_out = gmm.GaussianBp(pseudo, mu, sigma, grad_out)
        return pseudo_out, mu_out, sigma_out

def GmmConvStash(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
    return GmmStashFunction.apply(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma)

class GmmStashFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
        out, gauss = gmm.gmmconvstash(rowptr, colind, node_feat, pseudo, mu, sigma) # (E, K, F)
        ctx.save_for_backward(rowptr, colind, colptr, rowind, permute, gauss, node_feat, pseudo, mu, sigma)
        return out

    @staticmethod
    def backward(ctx, grad_out): # (E, F)
        rowptr, colind, colptr, rowind, permute, gauss, node_feat, pseudo, mu, sigma = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        guassian_grad = mhsddmm.mhsddmm(rowptr, colind, grad_out, node_feat)
        gauss_tran = mhtranspose.mhtranspose(permute, gauss) #
        node_feat_grad = mhspmm.mhspmm(colptr, rowind, gauss, grad_out)
        pseudo_out, mu_out, sigma_out = gmm.GaussianBp(pseudo, mu, sigma, guassian_grad)
        return None, None, None, None, None, node_feat_grad, pseudo_out, mu_out, sigma_out

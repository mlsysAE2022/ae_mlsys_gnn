from torch.utils.cpp_extension import load
import torch
import os

# path = os.path.join(os.path.dirname(__file__))
path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
path=os.path.join(path,'src')

u_add_v = load(
    name="u_add_v",
    sources=[os.path.join(path, "u_add_v/u_add_v.cpp"), os.path.join(path, "u_add_v/u_add_v.cu")],
    verbose=True,
)

def u_add_v_op(row_ptr,col_ind,row_val,col_val):
    return u_add_v_Function.apply(row_ptr,col_ind,row_val,col_val)
    

class u_add_v_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx,row_ptr,col_ind,row_val,col_val):
        result=u_add_v.u_add_v_forward(row_ptr,col_ind,row_val,col_val)
        ctx.save_for_backward(row_ptr,col_ind)
        return result
    
    @staticmethod
    def backward(ctx,grad_out):
        row_ptr,col_ind=ctx.saved_tensors
        grad_row,grad_col=u_add_v.u_add_v_backward(row_ptr,col_ind,grad_out)
        return None,None,grad_row,grad_col
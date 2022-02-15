from torch.utils.cpp_extension import load
import torch
import os

path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

transpose = load(
    name="csrtocsc",
    extra_cflags=["-lcusparse"],
    sources=[os.path.join(path, "src/csr2csc/csr2csc.cc"), os.path.join(path, "src/csr2csc/csr2csc.cu")],
    verbose=True,
)

def sparsetrans(rowptr, colind, numlist):
    _, _, out = transpose.csr2csc(rowptr, colind, numlist)
    return out
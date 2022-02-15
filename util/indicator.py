import os
import torch
from torch.utils.cpp_extension import load
import time

path = os.path.join(os.path.dirname(os.path.dirname(__file__)))

ind = load(
    name="ind",
    sources=[
        os.path.join(path, "src/util/indicator.cc"),
        os.path.join(path, "src/util/indicator.cu"),
    ],
    verbose=True,
)

def profile_start():
    return ind.start()

def profile_end():
    return ind.end()
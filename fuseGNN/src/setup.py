from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='fuseGNN',
    version='0.0.1',
    description='Custom library for graph convolutional networks for pytorch',
    author='Zhaodong Chen',
    author_email='chenzd15thu@ucsb.edu',
    ext_modules=[
        CUDAExtension('fgnn_agg', 
                      ['cuda/aggregate.cpp', 'cuda/aggregate_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']}),
        CUDAExtension('fgnn_format',
                      ['cuda/format.cpp', 'cuda/format_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70', '-lcusparse']}),
        CUDAExtension('fgnn_gcn',
                      ['cuda/gcn.cpp', 'cuda/gcn_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']}),
        CUDAExtension('fgnn_gat',
                      ['cuda/gat.cpp', 'cuda/gat_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']}),
        ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)

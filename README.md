# Getting Started

- Hardware

  - Intel CPU x86_64 with host memory 64GB. Tested on Intel Xeon Silver 4210 (10-core 20-thread) CPU with 512 GB host memory.
  - NVIDIA GPU with device memory 24GB. Tested on RTX3090 and RTX2080. We mainly evaluate our design on RTX3090 and the execution time may be different across different devices but the peak memory usage remains same.

- Software

  - PyTorch 1.8.0+
  - DGL 0.7.0+
  - Ninja 1.10+
  - GPUtil 1.4+

- OS&Compilation
  - Ubuntu 18.04+
  - CUDA 11.0+

# File Organization

- `src/`: the CUDA source code (`fused_edgeconv/fused_edgeconv.cu`,`fused_gat/fused_gat.cu`, and `fused_gmmconv/gmmconv.cu`) for GNN sparse kernels implemented with our proposed techniques, python binding of kernels (`fused_edgeconv/fused_edgeconv.cpp`, `fused_gat/fused_gat.cpp`, and `fused_gmmconv/gmmconv.cpp`), and some utilizations (`util/`).
- `operators/`: wrap our kernels into PyTorch modules.
- `layers/`: use the operators above to build up GAT, EdgeConv, and MoNet layers.
- `script/`: result analysis scripts to replicate our results.
- `example_data/`: raw experiment results.
- `docker/`: docker file for setting up the running environment.

# Installation

To build our software, you need to install Ninja and PyTorch as shown in dependencies.
We use the just-in-time compilation of the pytorch cpp-extension work flow.

> _Warning_
> If you find the bebug such as below, please comment the relevant code.

```
  File "/usr/local/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1606, in _get_cuda_arch_flags
    arch_list[-1] += '+PTX'
IndexError: list index out of range
```

Please make sure you have installed the nsight compute with your CUDA and configure your CUDA_HOME before replication. A typical path of nsight compute profiler is at `/usr/local/bin/nv-nsight-cu-cli`

# Docker

To avoid different path of CUDA installation, we provide a docker to run our implementation. You could run our AE code in our docker container below. We are sorry that profiling result needed by nv-nsight-cu-cli couldn’t be accomplished in the docker.

```bash
cd docker
docker build -t gnn:v1 -f Dockerfile .
docker run -it --runtime=nvidia --rm -v /home/yzm18/ae_mlsys_gnn/:/AE gnn:v1 /bin/bash
```

# Experiment Workflow

- Go to `script/` directory.
  First run `bash config.sh`.
- **Figure 7 result:** `./figure7.sh` to run end-to-end experiments on three GNN models. Generate `figure7.csv` and `figure7_io.csv`.
- **Figure 8 result:** `./figure8.sh` to run ablation study for operator reorganization. Generate `figure8.csv` and `figure8_io.csv`.
- **Figure 9 result:** `./figure8.sh` to run ablation study for operator fusion. Generate `figure9.csv` and `figure9_io.csv`.
- **Figure 10 result:** `./figure10.sh` to run ablation study for intermediate variable re-computation and generate `figure10.csv`.
- **Figure 11 result:** run `figure11_3090.sh` on RTX3090 and `figure11_2080.sh` on RTX2080. Generate `figure11_3090.csv` and `figure11_2080.csv`.
- **IO result:** run `./io.sh` to get all the IO results. Generate `figure7_io.csv`, `figure8_io.csv`, and `figure9_io.csv`.
- **fuseGNN result:** run `training_main.py` in the `gcnLib` submodule. Use the better result of `gas` and `gar` in `--mode`.

# Evaluation and Expected Result

Once you have run the experiment workflow, you can see the `.csv` result under the `script/` directory. The latency and memory results are stored in `figureX.csv`. The IO results can be seen in the corresponding `figureX_io.csv`.

Example output is given in the folder `example_data`.

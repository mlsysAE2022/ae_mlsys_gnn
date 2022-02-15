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

# Installation

To build our software, you need to install Ninja and PyTorch as shown in dependencies.
We use the just-in-time compilation of the pytorch cpp-extension work flow.

# Experiment Workflow

- Go to `script/` directory.
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

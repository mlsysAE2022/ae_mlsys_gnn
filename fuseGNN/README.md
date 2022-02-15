# fuseGNN

This is the source code for the paper "fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU".

## Organization of the project

* The "src" folder contains all the CUDA kernels we developed.
* The "fuseGNN" folder contains all the python APIs in different level: functional -> convs -> modules.

## Constraint random test

Under directory "fuseGNN/testbench", we provide two random testbenchs for a single layer GCN and GAT for both forward and backward passes.

## Training

The training of GCN and GAT on different datasets can be launched with "training_main.py". Different implementations can be selected with the argument "--mode", in particular, "geo" for pytorch geometric, gas for our fused-GAS abstraction, and gar for our fused-GAR abstraction. For GAT, we demonstrate the single attention head scenario, while the multi-attention head can be implemented by slightly modifying the related cuda kernels.

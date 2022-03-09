FROM nvidia/cuda:11.3.0-devel-ubuntu18.04 as base
FROM base as base-amd64

ENV NV_CUDNN_VERSION 8.2.0.53

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.3"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda11.3"
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} && \
    rm -rf /var/lib/apt/lists/*


# RUN dnf -y --disablerepo '*' --enablerepo=extras swap centos-linux-repos centos-stream-repos \
#     && dnf distro-sync \
#     && yum update -y

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get -y install tzdata

COPY install/ubuntu_build.sh /install/ubuntu_build.sh
RUN bash /install/ubuntu_build.sh

COPY install/ubuntu_conda.sh /install/ubuntu_conda.sh
RUN bash /install/ubuntu_conda.sh

COPY install/ubuntu_dgl.sh /install/ubuntu_dgl.sh
RUN bash /install/ubuntu_dgl.sh

# COPY install/conda_env/tensorflow_gpu.yml /install/conda_env/tensorflow_gpu.yml
# RUN ["/bin/bash", "-i", "-c", "conda env create -f /install/conda_env/tensorflow_gpu.yml"]

# COPY install/conda_env/mxnet_gpu.yml /install/conda_env/mxnet_gpu.yml
# RUN ["/bin/bash", "-i", "-c", "conda env create -f /install/conda_env/mxnet_gpu.yml"]

# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
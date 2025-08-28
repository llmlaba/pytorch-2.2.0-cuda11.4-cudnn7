# Ubuntu 20.04 + CUDA 11.4 + cuDNN 8 + toolchain
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ARG PYTORCH_BRANCH=release/2.2

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    TORCH_CUDA_ARCH_LIST="3.7" \ 
    USE_CUDA=1 

USER root

COPY test_torch.py /test_torch.py

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3-venv \
    git clinfo nvidia-utils-470 cmake-mozilla 

RUN python3 -m pip install --upgrade pip

WORKDIR /opt

RUN git clone --branch ${PYTORCH_BRANCH} https://github.com/pytorch/pytorch.git

WORKDIR /opt/pytorch

RUN python3 -m pip install -r requirements.txt

RUN python3 setup.py install

WORKDIR /opt

CMD ["python3", "/test_torch.py"]

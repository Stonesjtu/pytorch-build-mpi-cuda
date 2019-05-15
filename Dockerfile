FROM nvidia/cuda:9.0-devel-ubuntu16.04

ENV PYTORCH_VERSION=1.0.1
ENV CUDNN_VERSION=7.4.1.5-1+cuda9.0
ENV NCCL_VERSION=2.3.7-1+cuda9.0

ARG python=3.5
ENV PYTHON_VERSION=${python}

RUN apt update && apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
		build-essential \
		cmake \
		git \
		curl \
		vim \
		wget \
		ca-certificates \
		libcudnn7=${CUDNN_VERSION} \
		libnccl2=${NCCL_VERSION} \
		libnccl-dev=${NCCL_VERSION} \
		python${PYTHON_VERSION} \
		python${PYTHON_VERSION}-dev

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
	rm get-pip.py

# setup openmpi with CUDA and multi-threading support
WORKDIR "/workspace"

RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz && \
    gunzip -c openmpi-4.0.1.tar.gz | tar xf - && cd openmpi-4.0.1 && \
	mkdir build && cd build/ && \
	../configure --prefix=/usr --with-cuda --enable-mpi-thread-multiple && \
	make -j $(nproc) all && \
	make install && \
	ldconfig

# setup pytorch build dependencies
RUN pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

RUN git clone --branch v${PYTORCH_VERSION} --recursive https://github.com/pytorch/pytorch  && \
		cd pytorch && \
		python setup.py install

# setup apex
RUN git clone https://github.com/nvidia/apex && \
		cd apex && \
		pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# setup OpenSSH for MPI (should be deleted when FROM PHILLY CONTAINER)
RUN apt install -y --no-install-recommends openssh-client openssh-server && \
		mkdir -p /var/run/sshd

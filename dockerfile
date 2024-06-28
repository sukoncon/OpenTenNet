From pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Install Cutensor Python API
RUN mkdir /Ai4s/
COPY dependencies/cuTENSOR /workspace/cuTENSOR
COPY dependencies/libcutensor-linux-x86_64-1.7.0.1-archive /workspace/libcutensor-linux-x86_64-1.7.0.1-archive
ENV CUTENSOR_ROOT="/workspace/libcutensor-linux-x86_64-1.7.0.1-archive"
ENV PATH="/usr/local/cuda-12.1/bin:$PATH"
RUN cd /workspace/cuTENSOR/python && rm -rf build && pip install .

# # Install cuda kernel and python API
COPY submodule /workspace/Ai4S-release-submodule
ENV CUDA_HOME="/usr/local/cuda-12.1"
ENV TORCH_CUDA_ARCH_LIST="7.5+PTX"
RUN export TORCH_CUDA_ARCH_LIST="7.5+PTX" & cd /workspace/Ai4S-release-submodule/cuda \
&& rm -rf build && rm -rf *.egg-info && pip install .
RUN cd /workspace/Ai4S-release-submodule/python && rm -rf build && rm -rf __pycache__ && pip install .

# Install python pakages
RUN pip install py3nvml matplotlib 

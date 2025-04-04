# Base image with CUDA 12.0 + cuDNN 8
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

# Install Python and pip
RUN apt update && apt install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install JAX with CUDA 12 support
RUN pip install --upgrade pip
RUN pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Optional: install common ML libraries
RUN pip install numpy matplotlib pandas flax optax
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install mujoco
RUN pip install mujoco_mjx
RUN pip install brax==0.12.1
RUN pip install -q mediapy
RUN pip install tqdm
RUN pip install mediapy


## Set for headless renderring
# Install EGL & OpenGL libraries
RUN apt update && apt install -y \
    libegl1 \
    libgles2 \
    libgl1 \
    libx11-6 \
    libglfw3 \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev

# Install PyOpenGL + accelerate
RUN pip install PyOpenGL PyOpenGL_accelerate

# Ensure MUJOCO_GL is set to use EGL
ENV MUJOCO_GL=egl


RUN command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
# Set working directory inside the container
WORKDIR /workspace


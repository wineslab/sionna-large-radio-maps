FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 wget curl git && \
    wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/nvidia.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/ /" > /etc/apt/sources.list.d/nsight.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    build-essential \
    llvm \
    libegl1 \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxi6 \
    libxrandr2 \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
# Create and activate a virtual environment
WORKDIR /app
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Set environment variables for Sionna RT
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Core dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    h5py \
    scipy \
    pandas \
    tqdm \
    tensorflow[and-cuda]==2.19.*

# Geo/mapping dependencies
RUN pip3 install --no-cache-dir \
    geopandas \
    pyproj \
    shapely \
    osmnx \
    boto3

# 3D/mesh dependencies
RUN pip3 install --no-cache-dir \
    open3d \
    open3d-cpu \
    trimesh \
    triangle \
    pycollada

# Sionna RT (bundles mitsuba 3.5.2)
RUN pip3 install --no-cache-dir sionna-rt

# PIL for image processing (heightmap tiles)
RUN pip3 install --no-cache-dir pillow

# Set working directory
WORKDIR /app

# Default command
CMD ["python3", "scripts/gen_boulder_dataset.py"]

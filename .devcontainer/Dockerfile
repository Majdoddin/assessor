# Use the specified base image
FROM python:3.10.14-bookworm

# Set the working directory
WORKDIR /workspaces

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and PyTorch Audio
RUN pip install torch==2.0.0 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Copy the requirements.txt file into the image
RUN git clone -b assessor  https://github.com/Majdoddin/boolformer
RUN python3 -m venv .venv && . .venv/bin/activate && pip install -e /workspaces/boolformer

RUN git clone https://github.com/karpathy/nanoGPT.git

#COPY requirements.txt ./#



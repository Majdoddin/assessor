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
RUN pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Copy the requirements.txt file into the image
RUN git clone --mirror https://github.com/Majdoddin/boolformer
#RUN git clone --mirror https://github.com/Majdoddin/assessor.git
RUN git clone --mirror https://github.com/karpathy/nanoGPT.git

RUN mkdir assessor
#COPY requirements.txt ./#



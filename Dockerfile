FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System packages (optional: git, vim, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy project code into the image
COPY . /workspace

# Python dependencies (PyTorch is already provided by the base image)
RUN pip install --no-cache-dir \
    timm \
    mamba-ssm \
    einops \
    scikit-learn \
    matplotlib \
    scipy \
    pyyaml \
    tensorboard \
    wandb

ENV PYTHONUNBUFFERED=1

# Default command: show training options
CMD ["python", "trainv2_simple.py", "--help"]


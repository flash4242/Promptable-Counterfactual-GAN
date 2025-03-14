FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
ENV USERNAME=david
RUN pip install --upgrade pip

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    tmux \
    vim \
    htop \
    openssh-server \
    zip \
    unzip \
    build-essential \
    graphviz

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu118.html torch_geometric
RUN pip install debugpy pytest tensorboardX matplotlib seaborn pandas openpyxl wandb torchsummary scikit-learn

RUN useradd -m -s /bin/bash $USERNAME
WORKDIR /home/
USER $USERNAME

# Expose TensorBoard port
EXPOSE 6006
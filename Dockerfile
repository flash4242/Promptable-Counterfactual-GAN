FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
ENV USERNAME=david

# Install necessary packages
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
    graphviz \
    tree

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu118.html torch_geometric
RUN pip install debugpy pytest tensorboardX matplotlib seaborn pandas openpyxl wandb torchsummary scikit-learn imageio gradio
RUN pip install google.generativeai langchain-google-genai
ARG LANGCHAIN_VERSION=0.3.14
RUN pip install "langchain==${LANGCHAIN_VERSION}"


# Create a user with the correct UID and GID to avoid permission issues
ARG USER_ID=1002
ARG GROUP_ID=1002
RUN addgroup --gid $GROUP_ID $USERNAME && \
    adduser --disabled-password --gecos "" --uid $USER_ID --gid $GROUP_ID $USERNAME && \
    mkdir -p /home/$USERNAME && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME

# Set the user to avoid running as root
USER $USERNAME

# Set the environment variable for Matplotlib to avoid cache issues
ENV MPLCONFIGDIR=/tmp

# Set the working directory
WORKDIR /home/

# Expose TensorBoard port
EXPOSE 6006

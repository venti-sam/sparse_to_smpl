FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + Python + ROS2 prerequisites
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    software-properties-common \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Setup Locale for ROS2
RUN locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Add ROS2 Humble repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Minimal + RViz2
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-rviz2 \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace/amass

# Install PyTorch nightly (supports RTX 5080 Blackwell sm_120)
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other Python dependencies
RUN pip3 install --no-cache-dir \
    smplx[all] \
    numpy \
    scipy \
    tqdm \
    PyYAML \
    wandb

# Enable color terminal and source ROS2
ENV TERM=xterm-256color
RUN echo "export PS1='\[\e[1;32m\]\u@\h\[\e[m\]:\[\e[1;34m\]\w\[\e[m\]\$ '" >> /root/.bashrc \
    && echo "alias ls='ls --color=auto'" >> /root/.bashrc \
    && echo "alias grep='grep --color=auto'" >> /root/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Keep container running in the background
CMD ["tail", "-f", "/dev/null"]

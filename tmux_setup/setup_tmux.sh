#!/bin/bash

# Clone tmux plugins if they don't exist
if [ ! -d ~/.tmux/plugins/tpm ]; then
  git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
fi

if [ ! -d ~/.tmux/plugins/tmux-resurrect ]; then
  git clone https://github.com/tmux-plugins/tmux-resurrect ~/.tmux/plugins/tmux-resurrect
fi

if [ ! -d ~/.tmux/plugins/tmux-cpu ]; then
  git clone https://github.com/tmux-plugins/tmux-cpu ~/.tmux/plugins/tmux-cpu
fi

if [ ! -d ~/.tmux/plugins/vim-tmux-navigator ]; then
  git clone https://github.com/christoomey/vim-tmux-navigator ~/.tmux/plugins/vim-tmux-navigator
fi


# Stop on first error
set -e

# Determine the user's home directory inside the container
HOME_DIR=$(getent passwd $(whoami) | cut -d: -f6)
# Fallback if getent fails (rare, but possible)
if [ -z "$HOME_DIR" ]; then
  HOME_DIR="/root" # Usually root for a container if getent fails.
fi

# Set tmux directory (important in a container to avoid issues).
TMUX_DIR="$HOME_DIR/.tmux"

# Create the tmux directory if it doesn't exist
mkdir -p "$TMUX_DIR"

# Copy .tmux.conf to the user's home directory
cp .tmux.conf "$HOME_DIR/.tmux.conf"

# Change permissions if necessary (make sure the user can read the file)
chmod 644 "$HOME_DIR/.tmux.conf"

# Install tmux if it's not already installed.
# Adapt this based on your container's base image (e.g., apt-get, yum, apk)
if ! command -v tmux &> /dev/null; then
  echo "tmux not found. Installing..."
  # Example for Debian/Ubuntu:
  apt-get update && apt-get install -y tmux
  # Example for Alpine:
  # apk add tmux
  # Example for CentOS/RHEL:
  # yum install -y tmux
else
  echo "tmux is already installed."
fi

# Start tmux and source the config file. If a session exists, it just attaches to it
# Using tmux new-session -d to start it in detached mode if not already running to then use source file correctly.
# Check if a tmux server is already running
if tmux has-session >/dev/null 2>&1; then
    echo "tmux server is already running. Attaching..."
    tmux attach
else
    echo "Starting tmux and sourcing .tmux.conf..."
    tmux new-session -d  # Start detached so the script can source the config
    tmux source-file "$HOME_DIR/.tmux.conf"
    # tmux attach # Then immediately attach
fi

echo "tmux setup complete."


# Enable color prompt in .bashrc (if commented out)
if grep -q "^#force_color_prompt=yes" "$HOME_DIR/.bashrc"; then
  echo "Enabling color prompt in .bashrc..."
  sed -i "s/^#force_color_prompt=yes/force_color_prompt=yes/" "$HOME_DIR/.bashrc"
fi

# Reload .bashrc
source "$HOME_DIR/.bashrc"
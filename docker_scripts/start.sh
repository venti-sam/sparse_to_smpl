#!/bin/bash
set -e

# Get the absolute path to the project root directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Build the docker image
echo "Building amass_env docker image..."
docker build -t amass_env "$DIR"

# Run the docker container in detached mode
echo "Starting amass_env container..."
docker rm -f amass_container || true
docker run -d --name amass_container \
    --restart unless-stopped \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network host \
    -v "$DIR":/workspace/amass \
    -w /workspace/amass \
    amass_env

echo "Container started. Run ./docker_scripts/join.sh to access it."

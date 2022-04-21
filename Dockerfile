# EXAMPLE BUILD: docker build -t birdnet-analyzer -f ./Dockerfile .
# EXAMPLE RUN: docker run --rm --name birdnet-analyzer -v ~/Downloads:/audio birdnet-analyzer --i audio --o audio --slist audio

# Build from Python 3.8 slim
FROM python:3.8-slim

# Install required packages while keeping the image small
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg  && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install numpy scipy librosa tensorflow

# Neatly import all scripts
RUN mkdir -p /birdnet
WORKDIR /birdnet
COPY . ./

# Add entry point to run the script
ENTRYPOINT [ "python3", "./analyze.py" ]
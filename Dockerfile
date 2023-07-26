FROM python:3.9.17-slim-bullseye

# Install required Debian packages
RUN \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        ffmpeg \
    && \
    rm -rf /var/lib/apt/lists/*

# Upgrade Python packaging tools
RUN \
    pip3 install --no-cache-dir --upgrade \
        pip \
        setuptools \
        wheel

# Install required Python packages
RUN \
    pip3 install --no-cache-dir \
        bottle \
        librosa \
        numpy \
        scipy \
        resampy \
        tensorflow

WORKDIR /BirdNET
COPY . .

# Add entry point to run the script
ENTRYPOINT [ "python3" ]
CMD [ "birdnet/analysis/main.py" ]

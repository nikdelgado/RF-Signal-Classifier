# Use the official Python 3.8 image as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the local directory contents to the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        libfreetype6-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jupyter \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    tensorflow \
    keras \
    torch torchvision torchaudio

# Make port 8888 available outside this container
EXPOSE 8888

# Define environment variable
ENV NAME AI-ML-Development

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]


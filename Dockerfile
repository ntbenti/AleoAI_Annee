# Use an official PyTorch image as the base
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install necessary Python packages
RUN pip install --upgrade pip

# Install the required versions of Transformers and other packages
RUN pip install \
    transformers==4.33.0 \
    datasets==2.13.0 \
    accelerate==0.21.0 \
    bitsandbytes==0.42.0 \
    peft==0.5.0 \
    sagemaker-training

# Set environment variables
ENV SAGEMAKER_PROGRAM=train_codellama.py
ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_OUTPUT_DATA_DIR=/opt/ml/output/data

# Copy your training scripts and source code
COPY scripts/ /opt/ml/code/

# Set the working directory
WORKDIR /opt/ml/code/

# Define the entry point
ENTRYPOINT ["python", "train_codellama.py"]
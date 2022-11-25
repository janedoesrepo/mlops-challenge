FROM python:3.9.15-slim-buster

# Disable stdout and stderr buffer
ENV PYTHONUNBUFFERED 1

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir -r requirements.txt

# copy everything to the working directory
COPY scripts scripts
COPY test_images test_images
COPY trained_models trained_models

# run inference on test images on startup
CMD ["python", "scripts/main.py", "--ckpt_path", "trained_models/best_model.ckpt"]
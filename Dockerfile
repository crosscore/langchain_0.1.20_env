# Set the base image (host OS)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the content of the local directory to the working directory
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

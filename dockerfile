FROM python:3.9.16-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# working directory
WORKDIR /app

# Run the application
CMD ["bash"]
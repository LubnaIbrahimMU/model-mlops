# Use official Python slim image as base
FROM python:3.10-slim

# Set locale to UTF-8
ENV LANG=C.UTF-8

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install Docker inside the container
RUN apt-get update && apt-get install -y docker.io

# Install necessary packages including AWS CLI, Ansible, yq, and rsync
RUN apt-get update && apt-get install -y \
    openssh-client \
    curl \
    gnupg \
    unzip \
    sudo \
    awscli \
    ansible \
    rsync \
    jq \
    && pip install ansible boto3 botocore requests \
    # && ansible-galaxy collection install amazon.aws \
    && ansible-galaxy collection install community.aws --force \
    && ansible-galaxy collection install amazon.aws --force \
    && curl -sSL https://github.com/mikefarah/yq/releases/download/v4.6.1/yq_linux_amd64 -o /usr/bin/yq \
    && chmod +x /usr/bin/yq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the Ansible playbooks
WORKDIR /ansible

# Copy your Ansible playbooks and the encrypted Vault file to the container
COPY . /ansible

# Set Docker environment variables (optional, adjust if needed)
ENV DOCKER_HOST=unix:///var/run/docker.sock

# # Copy the script to decrypt the vault file
# COPY decrypt_vault.sh /usr/local/bin/decrypt_vault.sh
# RUN chmod +x /usr/local/bin/decrypt_vault.sh

# Set the entrypoint to the ansible-playbook command
ENTRYPOINT ["ansible-playbook"]
FROM heartexlabs/label-studio:0.9.0

# Install DVC and Git (needed for DVC)
RUN apt update -y -qq && \
    apt install -y -qq git rsync lsyncd procps curl unzip
RUN pip install -U --no-cache-dir \
    dvc==1.10.1 \
    pandas==1.0.5
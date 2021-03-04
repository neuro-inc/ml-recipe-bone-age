FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

COPY apt.txt .
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
        libsm6 \
        libxext6 \
        openssh-client \
        git \
        wget \
        unzip \
    && apt-get -qq clean \
    && apt-get -qq autoremove \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --progress-bar=off -r requirements.txt

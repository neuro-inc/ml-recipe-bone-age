FROM neuromation/base:v1.5

COPY apt.txt .
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    cat apt.txt | xargs -I % apt-get -qq install --no-install-recommends % && \
    apt-get -qq clean && \
    apt-get -qq autoremove && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --progress-bar=off -r requirements.txt

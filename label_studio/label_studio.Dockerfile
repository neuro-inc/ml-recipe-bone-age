FROM heartexlabs/label-studio:0.9.0

RUN apt-get update -y -qq --allow-releaseinfo-change && \
    apt-get install -y -qq --no-install-recommends \
        git \
        procps \
        openssh-client
RUN pip install -U --no-cache-dir \
    dvc[s3]==1.10.1

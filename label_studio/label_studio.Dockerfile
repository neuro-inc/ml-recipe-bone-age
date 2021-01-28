FROM heartexlabs/label-studio:0.9.0

RUN apt update -y -qq && \
    apt install -y -qq \
        git \
        procps \
        openssh-client
RUN pip install -U --no-cache-dir \
    dvc==1.10.1 \
    pandas==1.0.5

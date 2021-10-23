FROM python:3.7

RUN pip install --no-cache-dir \
    dvc[s3]==2.8.2 \
    pandas==1.2.1

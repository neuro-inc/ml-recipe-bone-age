FROM python:3.7

RUN pip install --no-cache-dir \
    dvc==1.11.13 \
    pandas==1.2.1

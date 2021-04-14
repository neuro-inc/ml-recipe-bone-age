FROM seldonio/seldon-core-s2i-python3:1.7.0-dev
ENV MODEL_NAME="seldon_deployment.seldon_model.SeldonModel" \
    API_TYPE="REST" \
    SERVICE_TYPE="MODEL" \
    PERSISTENCE="0"
# Copying in source code
COPY . /microservice
# Assemble script sourced from builder image based on user input or image metadata.
# If this file does not exist in the image, the build will fail.
RUN cd /microservice && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt
# Run script sourced from builder image based on user input or image metadata.
# If this file does not exist in the image, the build will fail.
CMD /s2i/bin/run

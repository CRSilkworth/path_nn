FROM tensorflow/tensorflow:2.1.1-gpu AS tf_build

WORKDIR /pipeline
COPY ./ ./
RUN python3 -m pip install tfx==0.21.4
RUN python3 -m pip freeze -l
RUN python3 -m pip install .

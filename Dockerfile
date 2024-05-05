# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
# FROM runpod/base:0.6.2-cuda12.1.0
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.


# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh
COPY ./data/06_models/tts_api/XTTS-v2/* ./data/06_models/tts_api/XTTS-v2
COPY ./data/06_models/tts_api/female_us_eng_johanna.mp3 ./data/06_models/tts_api/

# Python dependencies
COPY conf/base/runpod_tts/builder/requirements.txt /requirements.txt
# RUN python3.10 -m pip install --upgrade pip && \
#     # python3.10 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
#     python3.10 -m pip install --no-cache-dir runpod && \
#     python3.10 -m pip install --no-cache-dir torch TTS && \
#     rm /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir --user TTS
RUN rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.


# Add src files (Worker Template)
# ADD src .
ADD conf/base/runpod_tts/src .

CMD python -u /handler.py
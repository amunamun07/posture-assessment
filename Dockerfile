FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install build-essential -y

RUN mkdir -p /workspace/posture-assessment/
WORKDIR /workspace/posture-assessment/

#Installing Requirements
COPY ./setup.py .
RUN pip install .

COPY . .
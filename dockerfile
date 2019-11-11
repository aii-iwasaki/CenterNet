FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

RUN apt-get update
# for opencv
RUN apt-get install -y --no-install-recommends libsm6 libxext6 libxrender-dev libopencv-dev
# utils
RUN apt-get install -y --no-install-recommends git vim

RUN pip --no-cache-dir install Cython
RUN pip --no-cache-dir install pycocotools
RUN pip --no-cache-dir install matplotlib
RUN pip --no-cache-dir install opencv-python
RUN pip --no-cache-dir install numba
RUN pip --no-cache-dir install progress
RUN pip --no-cache-dir install easydict
RUN pip --no-cache-dir install scipy

COPY ./ /workspace/CenterNet/
RUN cd /workspace/CenterNet/models/networks/DCNv2/ && ./make.sh
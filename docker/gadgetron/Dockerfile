FROM gadgetron/ubuntu_1604_no_cuda

RUN apt-get update
RUN apt-get install -y software-properties-common python-software-properties
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN rm get-pip.py

VOLUME /opt/data
VOLUME /opt/work
ENV FASTMRI_DATA_DIR="/opt/data"

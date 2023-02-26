# FROM nvidia/cuda:10.2-base
FROM nvidia/cuda:11.0-base
 

CMD nvidia-smi

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

#set up environment
RUN apt update && apt install -y --no-install-recommends \
    tzdata \
    default-jdk \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    unzip

RUN pip3 -q install pip --upgrade

# install python environments 
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

RUN git clone https://github.com/vncorenlp/VnCoreNLP vncorenlp_data

#copies the applicaiton from local path to container path
COPY . /app

RUN cd /app/data && unzip zac2021-ltr-data.zip

CMD ["bash", "/app/scripts/run_train.sh", "&&", "bash", "/app/scripts/run_predict.sh"]
FROM  nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN apt update \
    && apt install -y \
    git\
    python3\
    python3-pip

RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /usr/local/src/*

ENV http_proxy=http://proxy:3128
ENV https_proxy=http://proxy:3128
ENV NODE_EXTRA_CA_CERTS=/usr/share/ca-certificates/extra/tri-ace-CA-2015.pem

COPY tri-ace-ca-2015.cer /usr/share/ca-certificates/extra/tri-ace-CA-2015.pem

RUN apt-get update -y && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/* && echo "extra/tri-ace-CA-2015.pem" >> /etc/ca-certificates.conf && update-ca-certificates

COPY requirements.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY *.py /home/
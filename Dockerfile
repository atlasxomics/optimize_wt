# Prologue
# DO NOT CHANGE
from 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:fe0b-main

workdir /tmp/docker-build/work/

shell [ \
    "/usr/bin/env", "bash", \
    "-o", "errexit", \
    "-o", "pipefail", \
    "-o", "nounset", \
    "-o", "verbose", \
    "-o", "errtrace", \
    "-O", "inherit_errexit", \
    "-O", "shift_verbose", \
    "-c" \
]
env TZ='Etc/UTC'
env LANG='en_US.UTF-8'

arg DEBIAN_FRONTEND=noninteractive

# Latch SDK
# DO NOT REMOVE
run pip install latch==2.76.5
run mkdir /opt/latch
run apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/latch/requirements.txt
RUN pip install --requirement /opt/latch/requirements.txt
RUN pip install https://github.com/atlasxomics/atx-common/archive/refs/tags/v0.1.0.tar.gz
RUN pip install --no-cache-dir setuptools
RUN pip install --no-cache-dir torch==2.8.0
RUN git clone https://github.com/RucDongLab/STAGATE_pyG.git /opt/STAGATE_pyG
RUN pip install --no-cache-dir torch-geometric==2.6.1
RUN pip install --no-cache-dir --no-binary=torch-scatter torch-scatter
RUN pip install --no-cache-dir --no-binary=torch-sparse torch-sparse
RUN pip install --no-cache-dir --no-binary=torch-cluster torch-cluster
RUN cd /opt/STAGATE_pyG && \
    python setup.py build && \
    python setup.py install

# Copy workflow data (use .dockerignore to skip files)
copy . /root/

# Epilogue

# Latch workflow registration metadata
# DO NOT CHANGE
arg tag
# DO NOT CHANGE
env FLYTE_INTERNAL_IMAGE $tag

workdir /root

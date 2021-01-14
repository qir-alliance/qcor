from ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive 
run apt-get -y update && apt-get install -y gcc g++ git wget gnupg lsb-release \
            libcurl4-openssl-dev python3 \
            libpython3-dev python3-pip libblas-dev ninja-build liblapack-dev \
    && python3 -m pip install ipopo cmake \
    && wget -qO- https://aide-qc.github.io/deploy/aide_qc/debian/PUBLIC-KEY.gpg | apt-key add - \
    && wget -qO- "https://aide-qc.github.io/deploy/aide_qc/debian/$(lsb_release -cs)/aide-qc.list" | tee -a /etc/apt/sources.list.d/aide-qc.list \
    && apt-get update \
    && apt-get install aideqc-llvm 

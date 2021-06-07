FROM codercom/code-server:latest
ENV DEBIAN_FRONTEND noninteractive
RUN echo 'deb http://deb.debian.org/debian testing main' | sudo tee /etc/apt/sources.list.d/testing.list
RUN sudo apt-get update \
 && sudo apt-get install -y \
    curl build-essential wget gdb \
    dumb-init \
    zsh \
    htop \
    locales \
    man \
    nano \
    git \
    procps \
    openssh-client \
    sudo \
    vim.tiny libssl-dev liblapack-dev libblas-dev ninja-build zlib1g-dev \
    libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev libbz2-dev \
    lsb-release libcurl4-openssl-dev \
  && sudo rm -rf /var/lib/apt/lists/* \
  && sudo wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz \
  && sudo tar xzf Python-3.8.2.tgz && cd Python-3.8.2 \
  && sudo CCSHARED='-fPIC' ./configure --prefix=/usr --exec-prefix=/usr --enable-shared --enable-optimizations \
  && sudo make altinstall && cd ../ && sudo rm -rf Python-3.8.2* \
  && wget -qO- https://aide-qc.github.io/deploy/aide_qc/debian/PUBLIC-KEY.gpg | sudo apt-key add - \
  && wget -qO- "https://aide-qc.github.io/deploy/aide_qc/debian/focal/aide-qc.list" | sudo tee -a /etc/apt/sources.list.d/aide-qc.list \
  && /usr/bin/pip3.8 install ipopo cmake --user \
  && sudo unlink /usr/bin/python3 && sudo ln -s /usr/bin/python3.8 /usr/bin/python3 \
  && sudo sed -i 's/#!\/usr\/bin\/python3/#!\/usr\/bin\/python3.9/' /usr/bin/lsb_release 

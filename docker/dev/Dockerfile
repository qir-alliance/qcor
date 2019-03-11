FROM theiaide/theia-full:next
USER root
RUN apt-get -y update \
    && apt-get -y update && apt-get install -y libcurl4-openssl-dev libssl-dev \
              python3 libpython3-dev python3-pip cmake gdb gfortran libblas-dev \
              liblapack-dev pkg-config
ADD settings.json /home/.theia/


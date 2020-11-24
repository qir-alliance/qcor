from qcor/deploy-base
workdir /home/dev
run cd /home/dev && apt-get update && apt-get install -y uuid-dev pkg-config ca-certificates \
    && git clone --recursive https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) install \
    && cd /home/dev && git clone https://github.com/ornl-qci/qcor && cd qcor && mkdir build && cd build \
    && cmake .. && make -j$(nproc) install \
    
# Theia application
workdir /home/dev
ARG version=latest
ADD $version.package.json ./package.json

RUN yarn --cache-folder ./ycache && rm -rf ./ycache && \
    NODE_OPTIONS="--max_old_space_size=4096" yarn theia build ;\
    yarn theia download:plugins
EXPOSE 3000
ENV SHELL=/bin/bash \
    THEIA_DEFAULT_PLUGINS=local-dir:/home/dev/plugins
ENV PYTHONPATH "${PYTHONPATH}:/root/.xacc:/root/.local/lib/python3.6/site-packages/psi4/lib"
ENV PATH "${PATH}:/root/.xacc/bin"

ENTRYPOINT [ "yarn", "theia", "start", "/home/dev", "--hostname=0.0.0.0" ]


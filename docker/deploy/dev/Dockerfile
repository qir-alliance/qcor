FROM xacc/deploy-base

RUN apt-get update && apt-get install -y lsb-release sudo && wget https://aide-qc.github.io/deploy/install.sh && bash install.sh 

ARG version=latest
ADD $version.package.json ./package.json

RUN yarn --cache-folder ./ycache && rm -rf ./ycache && \
    NODE_OPTIONS="--max_old_space_size=4096" yarn theia build ;\
    yarn theia download:plugins
EXPOSE 3000
ENV SHELL=/bin/bash \
    THEIA_DEFAULT_PLUGINS=local-dir:/home/dev/plugins
ENV PYTHONPATH "${PYTHONPATH}:/usr/local/xacc:/root/.local/lib/python3.6/site-packages/psi4/lib"

ENTRYPOINT [ "yarn", "theia", "start", "/home/dev", "--hostname=0.0.0.0" ]


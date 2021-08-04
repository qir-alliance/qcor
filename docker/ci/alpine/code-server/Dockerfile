FROM qcor/qcor-alpine
ENV VERSION=3.11.0

RUN apk add nodejs openssh-client gnupg bash sudo curl && \
   wget https://github.com/cdr/code-server/releases/download/v$VERSION/code-server-$VERSION-linux-amd64.tar.gz && \
   tar x -zf code-server-$VERSION-linux-amd64.tar.gz && \
   rm code-server-$VERSION-linux-amd64.tar.gz && \
   rm code-server-$VERSION-linux-amd64/node && \
   rm code-server-$VERSION-linux-amd64/code-server && \
   rm code-server-$VERSION-linux-amd64/lib/node && \
   mv code-server-$VERSION-linux-amd64 /usr/lib/code-server && \
   sed -i 's/"$ROOT\/lib\/node"/node/g'  /usr/lib/code-server/bin/code-server \
   && adduser --gecos '' --disabled-password coder \
   && echo "coder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd \
   && curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz" | tar -C /usr/local/bin -xzf - \
   && chown root:root /usr/local/bin/fixuid \
   && chmod 4755 /usr/local/bin/fixuid \
   && mkdir -p /etc/fixuid \
   && printf "user: coder\ngroup: coder\n" > /etc/fixuid/config.yml \
   && rm -rf fixuid-0.5-linux* \
   && ln -s /lib/libc.musl-x86_64.so.1 /lib/ld-linux-x86-64.so.2

USER 1000
ENV USER=coder
WORKDIR /home/coder
RUN git clone https://github.com/ornl-qci/qcor && cp -r qcor/examples cpp-examples \
   && cp -r qcor/python/examples py-examples && rm -rf qcor 

ENTRYPOINT ["/usr/lib/code-server/bin/code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none", "."]

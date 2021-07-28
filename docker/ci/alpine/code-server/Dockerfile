FROM qcor/qcor-alpine
ENV VERSION=3.11.0

RUN apk add nodejs openssh-client gnupg bash && \
   wget https://github.com/cdr/code-server/releases/download/v$VERSION/code-server-$VERSION-linux-amd64.tar.gz && \
   tar x -zf code-server-$VERSION-linux-amd64.tar.gz && \
   rm code-server-$VERSION-linux-amd64.tar.gz && \
   rm code-server-$VERSION-linux-amd64/node && \
   rm code-server-$VERSION-linux-amd64/code-server && \
   rm code-server-$VERSION-linux-amd64/lib/node && \
   mv code-server-$VERSION-linux-amd64 /usr/lib/code-server && \
   sed -i 's/"$ROOT\/lib\/node"/node/g'  /usr/lib/code-server/bin/code-server 

ENTRYPOINT ["/usr/lib/code-server/bin/code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none", "."]
# CMD ["--bind-addr 0.0.0.0:8080"]
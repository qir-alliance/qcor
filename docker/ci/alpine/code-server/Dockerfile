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
ADD README.md .
RUN sudo chown coder README.md && sudo chgrp coder README.md \
   && git clone https://github.com/ornl-qci/qcor && cp -r qcor/examples cpp-examples \
   && cp -r qcor/python/examples py-examples && rm -rf qcor \
   && mkdir -p /home/coder/.local/share/code-server/User \
   && printf "{\"workbench.startupEditor\": \"readme\", \"workbench.colorTheme\": \"Monokai Dimmed\", \"workbench.panel.defaultLocation\": \"right\", \"terminal.integrated.shell.linux\": \"bash\", \"files.associations\": {\"*.qasm\": \"cpp\"}" | tee /home/coder/.local/share/code-server/User/settings.json \
   && wget https://github.com/microsoft/vscode-cpptools/releases/download/1.5.1/cpptools-linux.vsix \
   && wget https://github.com/microsoft/vscode-python/releases/download/2020.10.332292344/ms-python-release.vsix \
   && /usr/lib/code-server/bin/code-server --install-extension cpptools-linux.vsix \
   && /usr/lib/code-server/bin/code-server --install-extension ms-python-release.vsix \
   && rm -rf cpptools-linux.vsix ms-python-release.vsix
   
ENV PATH "${PATH}:/usr/lib/code-server/bin"

ENTRYPOINT ["/usr/lib/code-server/bin/code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none", "."]

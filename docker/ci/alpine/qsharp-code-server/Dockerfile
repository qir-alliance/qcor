FROM qcor/cli
ENV VERSION=3.11.0

RUN sudo apk add nodejs openssh-client gnupg bash && \
   wget https://github.com/cdr/code-server/releases/download/v$VERSION/code-server-$VERSION-linux-amd64.tar.gz && \
   tar x -zf code-server-$VERSION-linux-amd64.tar.gz && \
   rm code-server-$VERSION-linux-amd64.tar.gz && \
   rm code-server-$VERSION-linux-amd64/node && \
   rm code-server-$VERSION-linux-amd64/code-server && \
   rm code-server-$VERSION-linux-amd64/lib/node && \
   sudo mv code-server-$VERSION-linux-amd64 /usr/lib/code-server && \
   sudo sed -i 's/"$ROOT\/lib\/node"/node/g'  /usr/lib/code-server/bin/code-server 

# WORKDIR /home/root

RUN sudo apk add bash icu-libs krb5-libs libgcc libintl libssl1.1 libstdc++ zlib wget \
   && sudo apk add libgdiplus --repository https://dl-3.alpinelinux.org/alpine/edge/testing/ \
   && wget https://dot.net/v1/dotnet-install.sh \
   && chmod +x dotnet-install.sh \
   && ./dotnet-install.sh -c 3.1 \
   && sudo ln -s ~/.dotnet/dotnet /usr/bin/dotnet \
   && dotnet nuget add source "https://pkgs.dev.azure.com/ms-quantum-public/Microsoft Quantum (public)/_packaging/alpha/nuget/v3/index.json" -n qdk-alpha \
   && dotnet new -i Microsoft.Quantum.ProjectTemplates \
   && git clone https://github.com/qir-alliance/qcor && cp -r qcor/examples cpp-examples && rm -rf qcor \
   && sudo apk add llvm \
   && sudo apk add xmlstarlet \
   && xmlstarlet ed --inplace -s /configuration -t elem -n config -v "" \
                              -s /configuration/config -t elem -n add -v "" \
                              -i /configuration/config/add -t attr -n "key" -v "maxHttpRequestsPerSource" \
                              -i /configuration/config/add -t attr -n "value" -v "2" \
                                 /home/coder/.nuget/NuGet/NuGet.Config \
   && sudo apk del xmlstarlet \  
   && dotnet new console && dotnet add package libllvm.runtime.ubuntu.20.04-x64 --version 11.0.0 \
   && cd /home/coder/.nuget/packages/libllvm.runtime.ubuntu.20.04-x64/11.0.0/runtimes/ubuntu.20.04-x64/native/ \
   && rm libLLVM.so \
   && ln -s /usr/lib/libLLVM-11.so libLLVM.so \
   && mkdir -p /home/coder/.local/share/code-server/User \
   && printf "{\"workbench.startupEditor\": \"readme\", \"workbench.colorTheme\": \"Monokai Dimmed\", \"workbench.panel.defaultLocation\": \"right\", \"terminal.integrated.shell.linux\": \"bash\", \"files.associations\": {\"*.qasm\": \"cpp\"}" | tee /home/coder/.local/share/code-server/User/settings.json \
   && wget https://github.com/microsoft/vscode-cpptools/releases/download/1.5.1/cpptools-linux.vsix \
   && wget https://github.com/microsoft/vscode-python/releases/download/2020.10.332292344/ms-python-release.vsix \
   && /usr/lib/code-server/bin/code-server --install-extension cpptools-linux.vsix \
   && /usr/lib/code-server/bin/code-server --install-extension ms-python-release.vsix \
   && rm -rf cpptools-linux.vsix ms-python-release.vsix
   
ENV PATH "${PATH}:/usr/lib/code-server/bin"
ADD README.md .
ENV QCOR_QDK_VERSION 0.17.2106148041-alpha
ENV LD_LIBRARY_PATH /home/coder/.nuget/packages/libllvm.runtime.ubuntu.20.04-x64/11.0.0/runtimes/ubuntu.20.04-x64/native

ENTRYPOINT ["/usr/lib/code-server/bin/code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none", "."]
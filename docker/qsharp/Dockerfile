FROM mcr.microsoft.com/dotnet/core/sdk:3.1-focal

# Mono is required to run pack.ps1, so we install it here.
RUN apt-get -y update && \
    apt-get -y install dirmngr gnupg apt-transport-https ca-certificates && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF && \
    sh -c 'echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" > /etc/apt/sources.list.d/mono-official-stable.list' && \
    apt-get -y update && \
    apt-get -y install mono-complete && \
    apt-get clean && rm -rf /var/lib/apt/lists/
# We can now get Mono itself.
RUN curl -o /usr/local/bin/nuget.exe https://dist.nuget.org/win-x86-commandline/latest/nuget.exe && \
    # Create as alias for nuget
    echo "alias nuget=\"mono /usr/local/bin/nuget.exe\"" >> /root/.bash_aliases

# Install qsharp compiler from source
RUN git clone https://github.com/microsoft/qsharp-compiler.git
RUN cd qsharp-compiler && export SOURCE_DIR=${PWD} && cd src/QsCompiler/CommandLineTool && dotnet build && \
    echo "export PATH=${PATH}:${SOURCE_DIR}/src/QsCompiler/CommandLineTool/bin/Debug/netcoreapp3.1/" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SOURCE_DIR}/src/QsCompiler/QirGeneration/Llvm.Net/runtimes/linux-x64/native" >> /root/.bashrc && \
    # Build QirGeneration extension
    cd ${SOURCE_DIR}/src/QsCompiler/QirGeneration && dotnet build && \
    # Export the DLL location to an environment variable.
    echo "export QIR_GEN_DLL=${SOURCE_DIR}/src/QsCompiler/QirGeneration/bin/Debug/netstandard2.1/Microsoft.Quantum.QirGeneration.dll" >> /root/.bashrc

RUN apt-get update && apt-get -y install cmake gcc g++ clang-format libcurl4-openssl-dev libunwind-dev libpython3-dev python3-pip libblas-dev liblapack-dev lsb-release && \
    wget -qO- https://aide-qc.github.io/deploy/aide_qc/debian/PUBLIC-KEY.gpg | apt-key add - && \
    wget -qO- "https://aide-qc.github.io/deploy/aide_qc/debian/$(lsb_release -cs)/aide-qc.list" | tee -a /etc/apt/sources.list.d/aide-qc.list && \
    apt-get update && apt-get -y install aideqc-llvm && \
    pip3 install cmake
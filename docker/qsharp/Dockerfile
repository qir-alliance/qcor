FROM mcr.microsoft.com/dotnet/core/sdk:3.1-focal

# Install Q# SDK
RUN dotnet new -i Microsoft.Quantum.ProjectTemplates

RUN apt-get update && apt-get -y install cmake gcc g++ clang-format libcurl4-openssl-dev libunwind-dev libpython3-dev python3-pip libblas-dev liblapack-dev lsb-release && \
    wget -qO- https://aide-qc.github.io/deploy/aide_qc/debian/PUBLIC-KEY.gpg | apt-key add - && \
    wget -qO- "https://aide-qc.github.io/deploy/aide_qc/debian/$(lsb_release -cs)/aide-qc.list" | tee -a /etc/apt/sources.list.d/aide-qc.list && \
    apt-get update && apt-get -y install aideqc-llvm && \
    pip3 install cmake
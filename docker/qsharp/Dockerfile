from qcor/deploy-base

# Install .NET
RUN wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && apt-get install -y apt-transport-https && apt-get update && apt-get install -y dotnet-sdk-3.1

# Install Q# SDK
# Add QDK-alpha source
RUN dotnet nuget add source "https://pkgs.dev.azure.com/ms-quantum-public/Microsoft Quantum (public)/_packaging/alpha/nuget/v3/index.json" -n qdk-alpha
RUN dotnet new -i Microsoft.Quantum.ProjectTemplates

# XACC and QCOR
RUN git clone --recursive https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) install 

RUN cd ../../ && git clone -b master https://github.com/ornl-qci/qcor && cd qcor && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc -DLLVM_ROOT=/usr/local/aideqc/llvm -DMLIR_DIR=/usr/local/aideqc/llvm/lib/cmake/mlir -DQCOR_BUILD_TESTS=TRUE \
    && make -j$(nproc) install && ctest --output-on-failure
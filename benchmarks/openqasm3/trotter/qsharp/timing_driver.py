# Timing the total Q# compilation workflow via QIR
# Reference: https://github.com/microsoft/qsharp-compiler/tree/main/examples/QIR/Optimization

import time, os, sys
os.chdir(sys.path[0])
start = time.time()
# Dotnet build -> generate ll file (unoptimized)
os.system('dotnet build')

# Optimize LLVM IR
os.system('/usr/local/aideqc/llvm/bin/clang -S qir/trotter.ll -O3 -emit-llvm -o qir/trotter-o3.ll')

# Compile and link:
os.system('/usr/local/aideqc/llvm/bin/llvm-as qir/trotter-o3.ll -o qir/trotter.bc')
os.system('/usr/local/aideqc/llvm/bin/llc -filetype=obj qir/trotter.bc -o qir/trotter.o')
os.system('/usr/local/aideqc/llvm/bin/clang++  -Wno-unused-command-line-argument -Wno-override-module -rdynamic -Wl,-rpath,/root/.xacc/lib:/root/.xacc/lib:/usr/local/aideqc/llvm/lib:/root/.xacc/clang-plugins -L /root/.xacc/lib -lqcor -lqrt -lqcor-jit -lqcor-quasimo -L /root/.xacc/lib -lxacc -lCppMicroServices -lxacc-quantum-gate -lxacc-pauli -lxacc-fermion -lpthread -lqir-qrt -D__internal__qcor__compile__backend="qpp" qir/trotter.o driver.cpp -o qir/a.out')
end = time.time()
print('Elapsed time:', end - start, ' [secs]')
# Clean up (no incremental builds)
os.system('rm -rf bin')
os.system('rm -rf obj')
os.system('rm -rf qir')

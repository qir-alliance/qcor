# Timing the total Q# compilation workflow via QIR
# Reference: https://github.com/microsoft/qsharp-compiler/tree/main/examples/QIR/Optimization

import time, os, sys
from statistics import mean, stdev

os.chdir(sys.path[0])

n_runs = 10

data = []
for run_id in range(n_runs): 
    start = time.time()
    # Dotnet build -> generate ll file (unoptimized)
    os.system('dotnet build')

    # Optimize LLVM IR
    os.system('/usr/local/aideqc/llvm/bin/clang -S qir/trotter.ll -O3 -emit-llvm -o qir/trotter-o3.ll')

    # Compile and link:
    os.system('/usr/local/aideqc/llvm/bin/llvm-as qir/trotter-o3.ll -o qir/trotter.bc')
    os.system('/usr/local/aideqc/llvm/bin/llc -filetype=obj qir/trotter.bc -o qir/trotter.o')
    os.system('/usr/local/aideqc/llvm/bin/clang++  -Wno-unused-command-line-argument -Wno-override-module -rdynamic -Wl,-rpath,/usr/local/aideqc/qcor/lib:/usr/local/aideqc/qcor/lib:/usr/local/aideqc/llvm/lib:/usr/local/aideqc/qcor/clang-plugins -L /usr/local/aideqc/qcor/lib -lqcor -lqrt -lqcor-jit -lqcor-quasimo -L /usr/local/aideqc/qcor/lib -lxacc -lCppMicroServices -lxacc-quantum-gate -lxacc-pauli -lxacc-fermion -lpthread -lqir-qrt -D__internal__qcor__compile__backend="qpp" qir/trotter.o driver.cpp -o qir/a.out')
    end = time.time()
    print('Elapsed time:', end - start, ' [secs]')
    data.append(end - start)
    # Clean up (no incremental builds)
    os.system('rm -rf bin')
    os.system('rm -rf obj')
    os.system('rm -rf qir')

print('==> Elapsed time =', mean(data), '+/-', stdev(data),  '[secs]')
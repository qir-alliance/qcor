# QIR as an IR enabling Quantum Language Integration
Here we demonstrate the utility of the QIR for enabling the integration of code from available quantum programming languages. Ideally, if one had quantum library code written in OpenQASM3, it should be usable / callable from Q# or QCOR, for example. The QIR makes this possible by lowering all program representations to a common representation, and letting existing linker tools provide integration across quantum language boundaries. 

## Goals

- Demonstrate the ability to program a quantum kernel in one language, and use from another. 

- Demonstrate the QCOR compiler and IR infrastructure as a mechanism for lowering languages to QIR and facilitating the linking phase to enable one language to invoke code from another.

- Demonstrate QCOR-calls-Q#

- Demonstrate QCOR-calls-OpenQASM3

- Demonstrate OpenQASM3-calls-Q#

- Demonstrate Q#-calls-OpenQASM3

- Demonstrate QCOR calls Qiskit

## Notes

```bash
export QCOR_QDK_VERSION=0.17.2106148041-alpha
export LD_LIBRARY_PATH=$HOME/.nuget/packages/libllvm.runtime.ubuntu.20.04-x64/11.0.0/runtimes/ubuntu.20.04-x64/native
```

## Outline

### Demo 1, QCOR Calls Q# QPE

- Walk through `qft.qs`, walk through `qpe.cpp`

- Compile it all together into a single executable / show with -v / run

```bash
qcor qft.qs qpe.cpp -shots 100 -v 
./a.out
```
- Show the generated QIR from Q#
```bash
code qir/qft.ll
```

### Demo 2, QCOR Calls OpenQASM3 QPE

- Note we could do the same thing with OpenQASM3, walk through both codes

- Compile it all together with -v / run
```bash
qcor iqft.qasm qpe.cpp -shots 100 -v
```
- Show the MLIR generated, show the LLVM QIR generated
```bash
qcor --emit-mlir iqft.qasm
qcor --emit-llvm iqft.qasm
```

### Demo 3, OpenQASM3 calls Q#, QRNG

- Show the Q# and QASM3 codes, talk through them

- Compile them together with -v / run
```bash
qcor qrng.qs driver.qasm -v 
./a.out
```
- Run a few times to show the randomness
```bash
for i in {1..10} ; do ./a.out ; done
``` 
- highlight the QIR and MLIR code
```bash
qcor --emit-mlir driver.qasm
qcor --emit-llvm driver.qasm
code qir/qrng.ll
```

### Demo 4, Q# call OpenQASM3

- Show the Q# and QASM3 code. Note the ability to pass Callables

- Compile with -v, run
```bash 
qcor op_takes_callable.qs kernel.qasm driver.cpp -v
./a.out
```

- Show the QIR generated
```bash
qcor --emit-mlir kernel.qasm
qcor --emit-llvm kernel.qasm
code qir/op_takes_callable.ll
```

### Demo 5, QCOR call Qiskit

- Show the python code, note how we map Qiskit to QIR, write to file
- Show the QCOR code, importing the correctly named function
- Compile and run with -v
```bash
python3 iqft.py
qcor -c iqft.ll -o iqft.o
qcor iqft.o -shots 100 qpe.cpp
```
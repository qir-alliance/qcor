# MLIR+QIR as an IR for Quantum-Classical Computing
Here we demonstrate the utility of the QIR and MLIR for enabling the development of compilers for available quantum languages. 

Our motivation with this demonstration is to show how MLIR+QIR enable the main feature of a robust IR: mapping multiple languages or programming approaches to multiple backends.

Our examples will be simple GHZ and Bell circuits for NISQ and FTQC execution, respectively. 

## Goals

- Demonstrate the utility of the MLIR and QIR for creating compilers and executable code for available quantum languages.

- Demonstrate MLIR as language-level IR for quantum-classical computing (control flow from Standard/Affine, etc.).

- Demonstrate write-once, run-on-any available quantum backend and multiple languages to multiple backends.

- Demonstrate accessibility of MLIR and QIR for available Pythonic circuit construction frameworks. 

## NOTES
Poor man's way to count instructions in the LLVM output
```bash
qcor --emit-llvm bell.qasm -O3 &> bell_tmp.ll && cat bell_tmp.ll | grep '%* = \|call' | wc -l && rm bell_tmp.ll
```

## Outline

### Demo 1, Simple Language Lowering, Helpful Classical Passes

- Show off the code in `ghz.qasm`. Note the gate function. 
- Goal is to highlight the unique benefits of mutliple levels of IR abstraction
- Compile and run, note we want to run this in NISQ mode
```bash
qcor -qrt nisq -shots 1000 ghz.qasm -o ghz.x
./ghz.x
```
- Run the compile command with `-v` to show the steps in the workflow
```bash
qcor -v ghz.qasm 
```
- Show the MLIR and QIR Output, noting the Affine Loop (and that we benefit from leveraging classical IR Dialects)
```bash
qcor --emit-mlir ghz.qasm
qcor --emit-llvm ghz.qasm
```
- Show simple classical optimizations, here inlining and loop unrolling
```bash
qcor --emit-milr -O3 ghz.qasm
qcor --emit-llvm -O3 ghz.qasm
```

- Show off the FTQC mode code, noting that the for loop has a conditional statement
- Compile and run with -v 
```bash 
qcor -qrt ftqc -v bell.qasm -o bell.x
./bell.x
```
- Show the MLIR and QIR
```bash
qcor --emit-mlir bell.qasm
qcor --emit-llvm bell.qasm
```
- Note the need for multiple layers of IR, can't get opts from MLIR, but can from LLVM
```bash
qcor --emit-mlir -O3 bell.qasm
qcor --emit-llvm -O3 llvm.qasm
qcor --emit-llvm bell.qasm -O0 &> bell_tmp.ll && cat bell_tmp.ll | grep '%* = \|call' | wc -l && rm bell_tmp.ll
qcor --emit-llvm bell.qasm -O3 &> bell_tmp.ll && cat bell_tmp.ll | grep '%* = \|call' | wc -l && rm bell_tmp.ll
```

- Switch to the Python script to show the audience how all the MLIR/QIR infrastructure just shown can also be generated from Python. Moreover, that Qiskit and Pyquil can also generate MLIR/QIR. 
- Note how `qjit` is the QCOR quantum just-in-time compiler, produces executable functions that enable `mlir()` and `llvm()` methods. 



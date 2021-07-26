# MLIR+QIR as an IR for Quantum-Classical Computing
Here we demonstrate the utility of the QIR and MLIR for enabling the development of compilers for available quantum languages. 

Our motivation with this demonstration is to show how MLIR+QIR enable the main feature of a robust IR: mapping multiple languages or programming approaches to multiple backends.

Our examples will be simple GHZ and Bell circuits for NISQ and FTQC execution, respectively. 

## Goals

- Demonstrate the utility of the MLIR and QIR for creating compilers and executable code for available quantum languages.

- Demonstrate MLIR as language-level IR for quantum-classical computing (control flow from Standard/Affine, etc.).

- Demonstrate write-once, run-on-any available quantum backend and multiple languages to multiple backends.

- Demonstrate accessibility of MLIR and QIR for available Pythonic circuit construction frameworks. 

## Outline

1. Show the GHZ QASM3 code. Note control flow, variable declarations, gate modifiers, etc.
2. Compile it to `ghz.x`. 
3. Immediately run it on perfect backend simulator.
4. Note we could just as easily run on physical backends. Kick of runs on IBM, IonQ, and Rigetti, all in separate terminals (set name of vscode terminal with 3 split panes).
5. While these are running, run on a separate new simulator (pick `aer`). Then run on noisy backend from `aer`. 
6. Now show how all this worked: 
    - compile the code again with `-v`, show the command line steps. 
    - Show the progressive lowering. Emit the MLIR code, show it off, its quantum-classical
    - Show the LLVM/QIR code. 
7. Now switch to `bell.qasm` and show how using the same language and IR, we can also run in FTQC mode (multi-modal IR and execution model). 
8. Switch to the Python script to show the audience how all the MLIR/QIR infrastructure just shown can also be generated from Python. Moreover, that Qiskit and Pyquil can also generate MLIR/QIR. Note how `qjit` is the QCOR quantum just-in-time compiler, produces executable functions that enable `mlir()` and `llvm()` methods. 
9. At any point, go back and check on the physical backend executions. 

## Notes:


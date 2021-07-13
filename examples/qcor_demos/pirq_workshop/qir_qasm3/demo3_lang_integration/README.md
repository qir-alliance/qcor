# QIR as an IR enabling Quantum Language Integration
Here we demonstrate the utility of the QIR for enabling the integration of code from available quantum programming languages. Ideally, if one had quantum library code written in OpenQASM3, it should be usable / callable from Q# or QCOR, for example. The QIR makes this possible by lowering all program representations to a common representation, and letting existing linker tools provide integration across quantum language boundaries. 

## Goals

- Demonstrate the ability to program a quantum kernel in one language, and use from another. 

- Demonstrate the QCOR compiler and IR infrastructure as a mechanism for lowering languages to QIR and facilitating the linking phase to enable one language to invoke code from another.

- Demonstrate QCOR-calls-Q#

- Demonstrate QCOR-calls-OpenQASM3

- Demonstrate OpenQASM3-calls-Q#

## Outline

1. Start off with the phase estimation example whereby qcor calls a Q# inverse quantum fourier transform function. Show off the code (both files), and `qir_nisq_kernel_utils.hpp` enabling one to decorate a Q# function with a qcor `QuantumKernel`.
2. Compile the code (use alpha version and in VERBOSE mode), passing the Q# file and the QCOR driver cpp file. Note the command line calls.
3. Run the executable, noting the correct result, and that the Q# code printed Hello World. 
4. Show the lowered `qft.qs` LLVM file, point to the `QCOR__IQFT__body` function. Note how in C++ we can mark a function as `extern`. 
5. Move to the qcor-to-qasm3 code...
6. ...
7. Move to the qasm3-calls-Q# code, a quantum random number generator in Q#, called from OpenQASM3 code. Note how we use `kernel` keyword in the language to mark function as `extern`. Show off the Q# code. 
8. Compile and run (run multiple times to show random numbers)


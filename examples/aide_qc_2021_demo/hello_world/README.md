# Hello World Demonstration

## GHZ Goals
Write a C++ code that prepares a GHZ state on N qubits. 
* How does one write a quantum kernel? 
* What is a qreg? 
* What operations are exposed on qreg? 
* How to reference qubits from a qreg? 
* Quantum Instruction Broadcasting
* Logical-to-physical connectivity mapping - placement
* Print the kernel qasm
* Run on Qpp, Noisy Aer, and IBM
* Demonstrate functional programming with lambdas
* Show equivalent in Python. First run is slow for JIT, but subsequently fast

## Circuit Optimization Goals
Write a C++ code that shows off a kernel that can be reduced to nothing by the Pass Manager.
* Loop over N Hadamards on a qubit
* Z H X H 
* Zero rotations
* Show off CX and X::ctrl 
* Show as_unitary_matrix
* Run with opt-level 1, then opt-level 2, -print-opt-stats
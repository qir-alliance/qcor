# Grover Demonstration
![circuit](grover_circuit.png)
Code up a grover example live, search for marked states |101> and |011>. We want a general library function that is parameterized on the 
oracle kernel and number of iterations

## Goals
* Show off kernel composition
* Show off compute-action-uncompute pattern
* Show off functional programming

## Steps
* Start off by hard-coding the oracle. And define run_grover to only take qreg and iterations
* Implement run_grover, using hard-coded oracle
* Implement amplification, showing circuit, showing pattern
* Implement main. 
* Compile and run. Show it works.

* Update run_grover to use KernelSignature, move oracle below run_grover.
* compile and run

* Update oracle to be a lambda

* Show off python version
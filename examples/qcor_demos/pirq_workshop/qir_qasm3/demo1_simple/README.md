# QASM3 MLIR and QIR Simple
Here we demonstrate the utility of the QIR and MLIR for enabling the development of compilers for available quantum languages. 
We show simple GHZ and Bell circuits for NISQ and FTQC execution, respectively. 

## Goals

- Demonstrate the utility of the MLIR and QIR for creating compilers and executable code for available quantum languages.

- Demonstrate write-once, run-on-any available quantum backend. 

- Demonstrate accessibility of MLIR and QIR for available Pythonic circuit construction frameworks. 

## Outline

GHZ (nisq) and Bell (ftqc) QASM3 codes, compile and run on simulator, IBM, Rigetti (in JupyterLab), IonQ. Walk through generated MLIR and QIR.

- Show pre-written GHZ and Bell QASM3 codes, note how GHZ is written for NISQ execution, while Bell is written for FTQC. 

- Lower GHZ to MLIR, show off the result. Lower GHZ to QIR, show off the result. 

- Lower Bell to MLIR, show off the result. Lower Bell to QIR, show off the result. 

- Compile and run the Bell code on QPP, note qcor -verbose and show the commands being run

- Compile the GHZ code for the IBM backend and execute. Note automated placement. 



## Notes:
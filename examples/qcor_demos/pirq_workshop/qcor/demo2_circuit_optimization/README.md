# QCOR Runtime Pass Manager Demonstration

## Test cases: circuits with optimization patterns (e.g., single-qubit sequences); Trotter circuits

## Goals

- QCOR runtime pass manager: circuit optimization (runtime, XACC IR level); optimization level, pass manager reporting.

- Topology mapping: backend configuration retrieval, noise-aware placement.

## Outline

- Circuit with simple gate sequences, e.g., H-Z-H-X; CLI option to enable optimization; pass information.

- Trotter circuit (e.g., 2-3 qubits): print gate count before and after optimization; examine the results (e.g. expectations and simulation runtime with and without optimization). 

- Topology placement: QAOA (max-cut) circuits => hardware mapping. 

- Noise-aware mapping: Bell test (H-CX); select an IBM backend with more qubits; should pick the pair with low CX error rate.




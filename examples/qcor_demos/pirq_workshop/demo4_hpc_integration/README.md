# QCOR HPC Simulator Backend

## Test cases: Sycamore random circuit sampling

## Goals

- Demonstrating system-level integration with HPC simulator backend (TNQVM/ExaTN on Summit) and remote Atos QLM.

- Configuring/targeting complex QPU backends via init files. 

## Outline

- Sycamore circuits: 53 qubits; print the circuit gate count, etc.

- Init file: configuring TNQVM (amplitude calculation mode). QCOR stack on Summit. Execution with MPI (showing GPU flops after calculation)

- Submit a job to QLM via a simple `-qpu` switch.
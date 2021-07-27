# Standard QCOR Workflow Demonstration

## Test case: GHZ (entangled) state generation and validation 

## Goals

- QCOR Intro: single-source quantum-classical programming; data types (qreg/AcceleratorBuffer); QCOR compiler CLI; QCOR -> XACC (runtime); XACC IR (Instruction/CompositeInstruction); Accelerator/QPU target (native gate set mapping).

- Topology placement and (runtime) circuit optimization passes (runtime pass manager)

- Advance: GHZ with many qubits (MPS simulation) and pulse-level IR (using IBM Accelerator in pulse mode)

## Outline

- Syntax-handling extension (`__qpu__`); `qreg` data type.

- Quantum kernel: execution (as a function call); kernel nesting; ctrl and adjoint modifiers; kernel IR printing (runtime).  

- CLI: compile (`-qpu` switch) 

- Execution: state-vector (`qpp`); noisy (`aer`); IBM; IonQ. Show QObj (IBMQ portal) to demonstate native gate set mapping (e.g. H -> rz and sx decomposition)

- Choose IBM pulse-mode (submitting pulses to IBM)

- Python binding: `ghz.py` - QCOR IR from Qiskit

- IonQ: Create `.ionq_config` with 

```
key: XXXX
url: url:https://api.ionq.co/v0.1
```
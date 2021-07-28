# Standard QCOR Workflow Demonstration

## Test case: GHZ (entangled) state generation and validation 

## Goals

- QCOR Intro: single-source quantum-classical programming; data types (qreg/AcceleratorBuffer); QCOR compiler CLI; QCOR -> XACC (runtime); XACC IR (Instruction/CompositeInstruction); Accelerator/QPU target (native gate set mapping).


## Outline

- Syntax-handling extension (`__qpu__`); `qreg` data type.

- Quantum kernel: execution (as a function call); kernel nesting; ctrl and adjoint modifiers; kernel IR printing (runtime).  

- CLI: compile (`-qpu` switch) 

- Execution: state-vector (`qpp`); noisy (`aer`); IBM; IonQ. Show QObj (IBMQ portal) to demonstate native gate set mapping (e.g. H -> rz and sx decomposition)

- IonQ: Create `.ionq_config` with 

```
key: XXXX
url: url:https://api.ionq.co/v0.1
```
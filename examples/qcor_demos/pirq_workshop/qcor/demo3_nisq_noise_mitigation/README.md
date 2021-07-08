# QCOR NISQ Noise Mitigation Demonstration

## Test cases: Identity circuit (no optimization) & VQE Deuteron H2
## Goals

- Demonstrating result post-processing (expectation calculation and noise mitigation).

- NISQ-era algorithms and utilities: QCOR optimizer, variational algorithms, noise mitigation, etc.

## Outline

- Circuit with an Identity sequence: repeating CNOT gates (no optimization), examining the expectation calculation (theoretical = 1.0) with and without noise mitigation. CLI option to enable noise mitigation on any backend (simulator and IBM)

- [Skipped] Sweeping VQE ansatz: showing the energy values with and without noise mitigation.

## Notes:

- Install mitiq: `pip3 install mitiq`

- Install Qiskit: `pip3 install qiskit`

- Make sure `~/.ibm_config` file is present.

- Noise model JSON generation 
(mitiq performs not very well with device noise model. Hence, use theoretical noise model for demonstration purposes)

```
from qiskit.providers.aer.noise import NoiseModel
import json 

# Use a depolarizing noise model.
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.001, 1),
    ["u1", "u2", "u3"],
)
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.01, 2),
    ["cx"],
)
  
print(json.dumps(noise_model.to_dict(True)))
```
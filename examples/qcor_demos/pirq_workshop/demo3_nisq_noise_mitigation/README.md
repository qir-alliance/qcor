# QCOR NISQ Noise Mitigation Demonstration

## Test cases: Identity circuit (no optimization) & VQE Deuteron H2
## Goals

- Demonstrating result post-processing (expectation calculation and noise mitigation).

- NISQ-era algorithms and utilities: QCOR optimizer, variational algorithms, noise mitigation, etc.

## Outline

- Circuit with an Identity sequence: repeating CNOT gates (no optimization), examining the expectation calculation (theoretical = 1.0) with and without noise mitigation. CLI option to enable noise mitigation on any backend (simulator and IBM)

- Sweeping VQE ansatz: showing the energy values with and without noise mitigation.
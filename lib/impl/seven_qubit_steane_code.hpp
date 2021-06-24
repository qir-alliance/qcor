#pragma once

// Distance-3 Steane quantum error correction code with 7 qubits
std::vector<std::vector<qcor::Operator>> seven_qubit_code_stabilizers() {
  static const std::vector<std::vector<qcor::Operator>> STABILIZERS{
      // Steane code has two groups of syndromes to detect X and Z errors.
      // X syndromes
      {qcor::X(0), qcor::X(2), qcor::X(4), qcor::X(6)},
      {qcor::X(1), qcor::X(2), qcor::X(5), qcor::X(6)},
      {qcor::X(3), qcor::X(4), qcor::X(5), qcor::X(6)},
      // Z syndromes
      {qcor::Z(0), qcor::Z(2), qcor::Z(4), qcor::Z(6)},
      {qcor::Z(1), qcor::Z(2), qcor::Z(5), qcor::Z(6)},
      {qcor::Z(3), qcor::Z(4), qcor::Z(5), qcor::Z(6)}};
  return STABILIZERS;
}

__qpu__ void seven_qubit_code_encoder(qreg q, int dataQubitIdx,
                                      std::vector<int> scratchQubitIdx) {
  H(q[scratchQubitIdx[0]]);
  H(q[scratchQubitIdx[2]]);
  H(q[scratchQubitIdx[5]]);
  CX(q[dataQubitIdx], q[scratchQubitIdx[4]]);
  CX(q[scratchQubitIdx[5]], q[scratchQubitIdx[1]]);
  CX(q[scratchQubitIdx[5]], q[scratchQubitIdx[3]]);
  CX(q[scratchQubitIdx[1]], q[dataQubitIdx]);
  CX(q[scratchQubitIdx[2]], q[scratchQubitIdx[4]]);
  CX(q[scratchQubitIdx[0]], q[scratchQubitIdx[4]]);
  CX(q[scratchQubitIdx[4]], q[scratchQubitIdx[5]]);
  CX(q[scratchQubitIdx[2]], q[scratchQubitIdx[3]]);
  CX(q[scratchQubitIdx[0]], q[scratchQubitIdx[1]]);
}

__qpu__ void seven_qubit_code_recover(qreg q, std::vector<int> logicalReg,
                                      std::vector<int> syndromes) {
  auto xSyndromes = {syndromes[0], syndromes[1], syndromes[2]};
  auto zSyndromes = {syndromes[3], syndromes[4], syndromes[5]};
  auto xSyndromeIdx = syndrome_array_to_int(xSyndromes);
  auto zSyndromeIdx = syndrome_array_to_int(zSyndromes);
  if (xSyndromeIdx > 0) {
    Z(q[xSyndromeIdx - 1]);
  }
  if (zSyndromeIdx > 0) {
    X(q[zSyndromeIdx - 1]);
  }
}

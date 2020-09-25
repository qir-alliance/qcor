#pragma once

// Distance-3 quantum error correction code with 5 qubits
std::vector<std::vector<qcor::PauliOperator>> five_qubit_code_stabilizers() {
  static const std::vector<std::vector<qcor::PauliOperator>> STABILIZERS{
      {qcor::X(0), qcor::Z(1), qcor::Z(2), qcor::X(3)},
      {qcor::X(1), qcor::Z(2), qcor::Z(3), qcor::X(4)},
      {qcor::X(0), qcor::X(2), qcor::Z(3), qcor::Z(4)},
      {qcor::Z(0), qcor::X(1), qcor::X(3), qcor::Z(4)}};
  return STABILIZERS;
}

__qpu__ void five_qubit_code_encoder(qreg q, int dataQubitIdx,
                                     std::vector<int> scratchQubitIdx) {
  CX(q[dataQubitIdx], q[scratchQubitIdx[1]);
  H(q[dataQubitIdx]);
  H(q[scratchQubitIdx[0]]);
  CX(q[dataQubitIdx], q[scratchQubitIdx[2]]);
  CX(q[scratchQubitIdx[0]], q[dataQubitIdx]);
  CX(q[dataQubitIdx], q[scratchQubitIdx[1]]);
  CX(q[scratchQubitIdx[0]], q[scratchQubitIdx[3]]);
  H(q[scratchQubitIdx[0]]);
  H(q[dataQubitIdx]);
  CX(q[scratchQubitIdx[0]], q[scratchQubitIdx[2]]);
  CX(q[dataQubitIdx], q[scratchQubitIdx[3]]);
  X(q[scratchQubitIdx[2]]);
}

__qpu__ void five_qubit_code_recover(qreg q, std::vector<int> logicalReg,
                                     std::vector<int> syndromes) {
  auto syndromeIndex = syndrome_array_to_int(syndromes);
  if (syndromeIndex > 0) {
    if (syndromeIndex == 1) {
      X(q[logicalReg[1]]);
    }
    if (syndromeIndex == 2) {
      Z(q[logicalReg[4]]);
    }
    if (syndromeIndex == 3) {
      X(q[logicalReg[2]]);
    }
    if (syndromeIndex == 4) {
      Z(q[logicalReg[2]]);
    }
    if (syndromeIndex == 5) {
      Z(q[logicalReg[0]]);
    }
    if (syndromeIndex == 6) {
      X(q[logicalReg[3]]);
    }
    if (syndromeIndex == 7) {
      Y(q[logicalReg[2]]);
    }
    if (syndromeIndex == 8) {
      X(q[logicalReg[0]]);
    }
    if (syndromeIndex == 9) {
      Z(q[logicalReg[3]]);
    }
    if (syndromeIndex == 10) {
      Z(q[logicalReg[1]]);
    }
    if (syndromeIndex == 11) {
      Y(q[logicalReg[1]]);
    }
    if (syndromeIndex == 12) {
      X(q[logicalReg[4]]);
    }
    if (syndromeIndex == 13) {
      Y(q[logicalReg[0]]);
    }
    if (syndromeIndex == 14) {
      Y(q[logicalReg[4]]);
    }
    if (syndromeIndex == 15) {
      Y(q[logicalReg[3]]);
    }
  }
}
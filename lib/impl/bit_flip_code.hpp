#pragma once

std::vector<std::vector<qcor::Operator>> bit_flip_code_stabilizers() {
  static const std::vector<std::vector<qcor::Operator>> STABILIZERS{
      {qcor::Z(0), qcor::Z(1)}, {qcor::Z(1), qcor::Z(2)}};
  return STABILIZERS;
}

__qpu__ void bit_flip_encoder(qreg q, int dataQubitIdx,
                              std::vector<int> scratchQubitIdx) {
  CX(q[dataQubitIdx], q[scratchQubitIdx[0]]);
  CX(q[dataQubitIdx], q[scratchQubitIdx[1]]);
}

__qpu__ void bit_flip_recover(qreg q, std::vector<int> logicalReg,
                              std::vector<int> syndromes) {
  const bool parity01 = (syndromes[0] == 1);
  const bool parity12 = (syndromes[1] == 1);
  // Correct error based on parity results
  if (parity01 && !parity12) {
    X(q[logicalReg[0]]);
  }

  if (parity01 && parity12) {
    X(q[logicalReg[1]]);
  }

  if (!parity01 && parity12) {
    X(q[logicalReg[2]]);
  }
}
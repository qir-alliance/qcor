#pragma once
#include <qcor_common>

std::vector<std::vector<qcor::PauliOperator>> bit_flip_code_stabilizers() {
  static const std::vector<std::vector<qcor::PauliOperator>> STABILIZERS{
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

#ifdef _QCOR_FTQC_RUNTIME
namespace ftqc {
__qpu__ void measure_stabilizer_generators(
    qreg q, std::vector<std::vector<qcor::PauliOperator>> stabilizerGroup,
    std::vector<int> logicalReg, int scratchQubitIdx,
    std::vector<int> &out_syndromes) {
  for (auto &stabilizer : stabilizerGroup) {
    for(auto &op : stabilizer) {
      // TODO: generalize for all codes
      std::map<int, int> bitMap;
      bitMap[0] = logicalReg[0];
      bitMap[1] = logicalReg[1];
      bitMap[2] = logicalReg[2];
      op.mapQubitSites(bitMap);
    }
    int syndromeResult;
    measure_basis_with_scratch(q, scratchQubitIdx, stabilizer, syndromeResult);
    out_syndromes.emplace_back(syndromeResult);
  }
}
} // namespace ftqc
#endif
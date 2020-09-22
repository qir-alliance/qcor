#pragma once
#include <qcor_common>

__qpu__ void bit_flip_encoder(qreg q, int dataQubitIdx,
                              std::vector<int> scratchQubitIdx) {
  CX(q[dataQubitIdx], q[scratchQubitIdx[0]]);
  CX(q[dataQubitIdx], q[scratchQubitIdx[1]]);
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
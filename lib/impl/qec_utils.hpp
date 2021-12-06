/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include <qcor_common>

int syndrome_array_to_int(const std::vector<int> &syndromes) {
  int result = 0;
  for (int i = 0; i < syndromes.size(); ++i) {
    result += ((1 << i) * syndromes[i]);
  }
  return result;
}

#ifdef _QCOR_FTQC_RUNTIME
namespace ftqc {
__qpu__ void measure_stabilizer_generators(
    qreg q, std::vector<std::vector<qcor::Operator>> stabilizerGroup,
    std::vector<int> logicalReg, int scratchQubitIdx,
    std::vector<int> &out_syndromes) {
  for (auto &stabilizer : stabilizerGroup) {
    for (auto &op : stabilizer) {
      std::map<int, int> bitMap;
      for (int i = 0; i < logicalReg.size(); ++i) {
        bitMap[i] = logicalReg[i];
      }
      op.mapQubitSites(bitMap);
    }
    int syndromeResult;
    measure_basis_with_scratch(q, scratchQubitIdx, stabilizer, syndromeResult);
    out_syndromes.emplace_back(syndromeResult);
  }
}
} // namespace ftqc
#endif
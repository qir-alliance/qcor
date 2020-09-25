
#pragma once
#include <functional>
#include "bit_flip_code.hpp"
#include "five_qubit_code.hpp"
#include "seven_qubit_steane_code.hpp"

// Encode a physical qubit into a logical qubit using given scratch qubits.
// Inputs:
// - Qubit register
// - Data qubit index (int)
// - List of scratch qubits (vector<int>)
using encodeFn = std::function<void(qreg, int, std::vector<int>)>;
// Recover (apply error correction) based on syndrome data:
// Inputs:
// - Qubit register
// - Logical qubit register (vector<int>)
// - Syndrome data (vector<int>)
using recoverFn = std::function<void(qreg, std::vector<int>, std::vector<int>)>;

// Set of stabilizers define the QEC code
using stabilizerGroups = std::vector<std::vector<qcor::PauliOperator>>;

using QecCode = std::tuple<stabilizerGroups, encodeFn, recoverFn>;

// Get the set of utils to handle a specific QEC code
// Code name:
// - "bit-flip"
// - "five-qubit"
// - "steane" (7 qubits)
// - "surface-code",
// etc.
QecCode getQecCode(const std::string &in_codeName) {
  if (in_codeName == "bit-flip") {
    encodeFn encoder(
        [](qreg q, int dataQubitIdx, std::vector<int> scratchQubitIdx) {
          bit_flip_encoder(q, dataQubitIdx, scratchQubitIdx);
        });
    recoverFn recover(
        [](qreg q, std::vector<int> logicalReg, std::vector<int> syndromes) {
          bit_flip_recover(q, logicalReg, syndromes);
        });
    return std::make_tuple(bit_flip_code_stabilizers(), encoder, recover);
  }
  if (in_codeName == "five-qubit") {
    encodeFn encoder(
        [](qreg q, int dataQubitIdx, std::vector<int> scratchQubitIdx) {
          five_qubit_code_encoder(q, dataQubitIdx, scratchQubitIdx);
        });
    recoverFn recover(
        [](qreg q, std::vector<int> logicalReg, std::vector<int> syndromes) {
          five_qubit_code_recover(q, logicalReg, syndromes);
        });
    return std::make_tuple(five_qubit_code_stabilizers(), encoder, recover);
  }
  if (in_codeName == "steane") {
    encodeFn encoder(
        [](qreg q, int dataQubitIdx, std::vector<int> scratchQubitIdx) {
          seven_qubit_code_encoder(q, dataQubitIdx, scratchQubitIdx);
        });
    recoverFn recover(
        [](qreg q, std::vector<int> logicalReg, std::vector<int> syndromes) {
          seven_qubit_code_recover(q, logicalReg, syndromes);
        });
    return std::make_tuple(seven_qubit_code_stabilizers(), encoder, recover);
  }
  
  throw std::runtime_error("Error: '" + in_codeName +
                           "' is not a valid QEC code.");
}
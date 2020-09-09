#include <qalloc>

// Example demonstrates a simple 3-qubit bit-flip code.
// Compile:
// qcor -qpu aer[noise-model:<noise.json>] -qrt ftqc bit-flip-code.cpp 
// ./a.out

bool getLogicalVal(qreg q, int logicalIdx) {
  if (q.creg[logicalIdx*3] == q.creg[logicalIdx*3]) {
    // First two bits matched.
    return q.creg[logicalIdx*3];
  }
  // The last bit is the tie-breaker.
  return q.creg[logicalIdx*3 + 2];
}

// Encode qubits into logical qubits:
// Assume q[0], q[3], q[6] are initial physical qubits
// that will be mapped to logical qubits q[0-2], q[3-5], etc.
__qpu__ void encodeQubits(qreg q) {
  int nbLogicalQubits = q.size() / 3;
  for (int i = 0; i < nbLogicalQubits; ++i) {
    int physicalQubitIdx = 3 * i;
    CX(q[physicalQubitIdx], q[physicalQubitIdx + 1]);
    CX(q[physicalQubitIdx], q[physicalQubitIdx + 2]);
  }
}

// Logical CNOT b/w two logical qubits
__qpu__ void cnotLogical(qreg q) {
  for (int i = 0; i < 3; ++i) {
    CX(q[i], q[3 + i]);
  }
}

__qpu__ void measureLogical(qreg q, int logicalIdx) {
  int physicalIdx = logicalIdx * 3;
  for (int i = physicalIdx; i < physicalIdx + 3; ++i) {
    Measure(q[i]);
  }
}

__qpu__ void correctLogicalQubit(qreg q, int logicalIdx, int ancIdx) {
  int physicalIdx = logicalIdx * 3;
  // Assume that we only has 1 ancilla qubit
  CX(q[physicalIdx], q[ancIdx]);
  CX(q[physicalIdx + 1], q[ancIdx]);
  const bool parity01 = Measure(q[ancIdx]);
  if (parity01) {
    // Reset anc qubit for reuse
    X(q[ancIdx]);
  }

  CX(q[physicalIdx + 1], q[ancIdx]);
  CX(q[physicalIdx + 2], q[ancIdx]);
  const bool parity12 = Measure(q[ancIdx]);
  if (parity12) {
    // Reset anc qubit
    X(q[ancIdx]);
  }

  // Correct error based on parity results
  if (parity01 && !parity12) {
    X(q[physicalIdx]);
  }  
  
  if (parity01 && parity12) {
    X(q[physicalIdx + 1]);
  }  

  if (!parity01 && parity12) {
    X(q[physicalIdx + 2]);
  }
}

__qpu__ void runQecCycle(qreg q) {
  int nbLogicalQubits = q.size() / 3;
  int ancBitIdx = q.size() - 1;
  for (int i = 0; i < nbLogicalQubits; ++i) {
    correctLogicalQubit(q, i, ancBitIdx);
  }
}

__qpu__ void resetAll(qreg q) {
  for (int i = 0; i < q.size(); ++i) {
    // Reset qubits by measure + correct.
    if (Measure(q[i])) {
      X(q[i]);
    }
  }
}

// Error corrected Bell example:
// Note: the 3-q bit-flip code can only protect against X errors.
__qpu__ void bellQEC(qreg q, int nbRuns) {
  using qcor::xasm;
  int ancQbId = 6;
  for (int i = 0; i < nbRuns; ++i) {
    // Apply H before encoding.
    H(q[0]);
    // Encode the qubits into logical qubits.
    encodeQubits(q);
    // Run a QEC cycle
    runQecCycle(q);
    // Apply *logical* CNOT
    cnotLogical(q);
    // Run a QEC cycle
    runQecCycle(q);
    
    // Measure *logical* qubits
    measureLogical(q, 0);
    measureLogical(q, 1);

    // Get *logical* results
    const bool logicalReq0 = getLogicalVal(q, 0);
    const bool logicalReq1 = getLogicalVal(q, 1);
    
    if (logicalReq0 == logicalReq1) {
      std::cout << "Iter " << i << ": Matched!\n";
    } else {
      std::cout << "Iter " << i << ": NOT Matched!\n";
    }
    resetAll(q);
  }
}

int main() {
  // Note: we need 3 physical qubits for each logical qubit +
  // an ancilla qubit for syndrom measurement.
  auto q = qalloc(7);
  bellQEC(q, 100);
}
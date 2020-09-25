#include <qcor_qec>

// Note: although QEC encode and syndrome decode kernels don't need FTQC
// runtime, in order to feed-forward syndrome, we need to use FTQC runtime.
// Compile with:
// $ qcor -qrt ftqc -qpu qpp five_qubit_qec_code.cpp

// Helper kernel to verify error-correcting procedure
__qpu__ void applyError(qreg q, int qIdx, int opId) {
  if (opId == 1) {
    std::cout << "Apply X error @ q[" << qIdx << "].\n";
    X(q[qIdx]);
  }
  if (opId == 2) {
    std::cout << "Apply Y error @ q[" << qIdx << "].\n";
    Y(q[qIdx]);
  }
  if (opId == 3) {
    std::cout << "Apply Z error @ q[" << qIdx << "].\n";
    Z(q[qIdx]);
  }
}

using namespace ftqc;

int main() {
  // Five-qubit code + 1 scratch qubit for syndrome measurement
  auto q = qalloc(6);
  const std::vector<int> LOGICAL_REG{0, 1, 2, 3, 4};
  const int ANC_QUBIT = 5;

  // Retrieve "five-qubit" code:
  auto [stabilizers, encodeFunc, recoverFunc] = getQecCode("five-qubit");
  // Encode the qubit: from qubit 0 to qubit register [0-4]
  encodeFunc(q, 0, {1, 2, 3, 4});

  // Test all possible *single-qubit* error
  for (int qId = 0; qId < 5; ++qId) {
    for (int opId = 1; opId <= 3; ++opId) {
      // If using a perfect simulator, apply a random error to observe syndrome
      // changes.
      applyError(q, qId, opId);
      std::vector<int> syndromes;
      // Measure the stabilizer syndromes:
      measure_stabilizer_generators(q, stabilizers, LOGICAL_REG, ANC_QUBIT,
                                    syndromes);
      assert(syndromes.size() == 4);
      std::cout << "Syndrome: <X0Z1Z2Z3> = " << syndromes[0]
                << "; <X1Z2Z3X4> = " << syndromes[1]
                << "; <X0X2Z3Z4> = " << syndromes[2]
                << "; <Z0X1X3Z4> = " << syndromes[3] << "\n";

      // Recover:
      recoverFunc(q, LOGICAL_REG, syndromes);
      // Measure again to check:
      syndromes.clear();
      measure_stabilizer_generators(q, stabilizers, LOGICAL_REG, ANC_QUBIT,
                                    syndromes);
      std::cout << "AFTER CORRECTION: \nSyndrome: <X0Z1Z2Z3> = " << syndromes[0]
                << "; <X1Z2Z3X4> = " << syndromes[1]
                << "; <X0X2Z3Z4> = " << syndromes[2]
                << "; <Z0X1X3Z4> = " << syndromes[3] << "\n";
    }
  }
}
#include <qcor_qec>

// Note: although QEC encode and syndrome decode kernels don't need FTQC
// runtime, in order to feed-forward syndrome, we need to use FTQC runtime.
// Compile with:
// $ qcor -qrt ftqc -qpu qpp steane_qec_code.cpp

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
  // Steane is a  seven-qubit code + 1 scratch qubit for syndrome measurement
  auto q = qalloc(8);
  const std::vector<int> LOGICAL_REG{0, 1, 2, 3, 4, 5, 6};
  const int ANC_QUBIT = 7;

  // Retrieve "steane" code:
  auto [stabilizers, encodeFunc, recoverFunc] = getQecCode("steane");
  // Encode the qubit: from qubit 0 to qubit register [0-6]
  encodeFunc(q, 0, {1, 2, 3, 4, 5, 6});

  // Test all possible *single-qubit* error
  for (int opId = 1; opId <= 3; ++opId) {
    for (int qId = 0; qId < 7; ++qId) {
      // If using a perfec simulator, apply a random error to observe syndrome
      // changes.
      applyError(q, qId, opId);
      std::vector<int> syndromes;
      // Measure the stabilizer syndromes:
      measure_stabilizer_generators(q, stabilizers, LOGICAL_REG, ANC_QUBIT,
                                    syndromes);
      assert(syndromes.size() == 6);
      const auto printSyndrome = [](const std::vector<int> &syndVals) {
        // First 3 are X syndromes
        std::cout << "X syndrome: ";
        for (int i = 0; i < 3; ++i) {
          std::cout << syndVals[i] << " ";
        }
        // Next 3 are Z syndromes
        std::cout << "\nZ syndrome: ";
        for (int i = 3; i < 6; ++i) {
          std::cout << syndVals[i] << " ";
        }
        std::cout << "\n";
      };

      printSyndrome(syndromes);
      // Recover:
      recoverFunc(q, LOGICAL_REG, syndromes);
      // Measure again to check:
      syndromes.clear();
      measure_stabilizer_generators(q, stabilizers, LOGICAL_REG, ANC_QUBIT,
                                    syndromes);
      std::cout << "AFTER CORRECTION: \n";
      printSyndrome(syndromes);
    }
  }
}
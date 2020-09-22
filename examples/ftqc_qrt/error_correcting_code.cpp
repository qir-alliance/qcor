#include <qcor_qec>

__qpu__ void applyError(qreg q, int qIdx) {
  std::cout << "Apply X error @ q[" << qIdx << "].\n";
  X(q[qIdx]);
}

using namespace ftqc;

int main() {
  auto q = qalloc(4);
  bit_flip_encoder(q, 0, {1, 2});
  std::vector<int> syndromes;
  // If using a perfec simulator, apply a random X error to observe syndrome changes.
  applyError(q, 2);

  // Measure the stabilizer syndromes:
  measure_stabilizer_generators(q, bit_flip_code_stabilizers(), {0, 1, 2}, 3, syndromes);
  assert(syndromes.size() == 2);
  std::cout << "Syndrome: <Z0Z1> = " << syndromes[0] << "; <Z1Z2> = " << syndromes[1] << "\n";
  
  // Recover:
  bit_flip_recover(q, {0, 1, 2}, syndromes);
  // Measure again to check:
  syndromes.clear();
  measure_stabilizer_generators(q, bit_flip_code_stabilizers(), {0, 1, 2}, 3, syndromes);
  std::cout << "AFTER CORRECTION: \nSyndrome: <Z0Z1> = " << syndromes[0] << "; <Z1Z2> = " << syndromes[1] << "\n";
}
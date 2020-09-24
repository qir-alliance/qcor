#include <qcor_qec>

__qpu__ void applyError(qreg q, int qIdx) {
  std::cout << "Apply X error @ q[" << qIdx << "].\n";
  X(q[qIdx]);
}

using namespace ftqc;

int main() {
  auto q = qalloc(4);
  // Retrieve "bit-flip" code:
  auto [stabilizers, encodeFunc, recoverFunc] = getQecCode("bit_flip");
  encodeFunc(q, 0, {1, 2});  
  // If using a perfect simulator, apply a random X error to observe syndrome changes.
  applyError(q, 0);
  
  std::vector<int> syndromes;
  // Measure the stabilizer syndromes:
  measure_stabilizer_generators(q, stabilizers, {0, 1, 2}, 3, syndromes);
  assert(syndromes.size() == 2);
  std::cout << "Syndrome: <Z0Z1> = " << syndromes[0] << "; <Z1Z2> = " << syndromes[1] << "\n";
  
  // Recover:
  recoverFunc(q, {0, 1, 2}, syndromes);
  // Measure again to check:
  syndromes.clear();
  measure_stabilizer_generators(q, stabilizers, {0, 1, 2}, 3, syndromes);
  std::cout << "AFTER CORRECTION: \nSyndrome: <Z0Z1> = " << syndromes[0] << "; <Z1Z2> = " << syndromes[1] << "\n";
}
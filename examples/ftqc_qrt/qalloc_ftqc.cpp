#include <qalloc>
// Compile with: qcor -qpu qpp -qrt ftqc qalloc_ftqc.cpp

__qpu__ void test(qreg q, std::vector<int> &result, int shots) {
  for (int i = 0; i < shots; ++i) {
    // Allocate inside a big loop to make sure
    // ancilla qubits are reused appropriately.
    auto anc_reg = qalloc(1);
    H(q[0]);
    CX(q[0], anc_reg[0]);
    int value = 0;
    if (Measure(q[0])) {
      value = value + 1;
      X(q[0]);
    }
    if (Measure(anc_reg[0])) {
      value = value + 2;
      X(anc_reg[0]);
    }
    result.emplace_back(value);
  }
}

int main() {
  auto q = qalloc(1);
  std::vector<int> results;
  test(q, results, 1024);

  int count00 = 0;
  int count11 = 0;
  for (const auto &result : results) {
    if (result == 0) {
      count00++;
    }
    if (result == 3) {
      count11++;
    }
  }
  std::cout << "Count 00 = " << count00 << "; Count 11 = " << count11 << "\n";
  // Reasonable balance Bell distribution
  qcor_expect(count00 + count11 == 1024);
  qcor_expect(count00 > 400);
  qcor_expect(count11 > 400);
}

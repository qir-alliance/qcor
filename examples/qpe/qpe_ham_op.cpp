#include <qcor_qpe>


/// Compile with:
/// qcor -qpu qpp qpe_ham_op.cpp

int main(int argc, char **argv) {
  auto q = qalloc(3);
  auto H = X(0) * X(1);
  int k = 1;
  double omega = 1.234;
  const int num_time_slices = 2;
  std::cout << "Kernel:\n";
  pauli_qpe_iter::print_kernel(std::cout, q, k, omega, H, num_time_slices, true);
}

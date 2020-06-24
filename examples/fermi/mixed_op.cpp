#include "qcor.hpp"

using namespace qcor;

__qpu__ void ansatz(qreg q, double theta, std::shared_ptr<xacc::Observable> H) {
  X(q[0]);
  X(q[1]);
}

int main(int argc, char **argv) {
  using qcor::X; // Due to ambiguous call with xacc::quantum::X

  auto q = qalloc(2);

  std::cout << "Fermi op " << std::endl;

  auto fermi_H = adag(0) * a(1) + adag(1) * a(0);

  std::cout << fermi_H.toString() << std::endl;

  std::cout << "fermi -> pauli transf " << std::endl;

  auto pauli_H = transform(fermi_H);

  std::cout << pauli_H.toString() << std::endl;

  std::cout << "mixed sum: " << std::endl;

  auto mixed_sum = fermi_H + pauli_H;

  std::cout << mixed_sum.toString() << std::endl;

  std::cout << "mixed product " << std::endl;

  auto mixed_prod = adag(0) * X(1) + X(0) * a(1);

  std::cout << mixed_prod.toString() << std::endl;

  auto H = X(0) + X(0) * a(1) + adag(1) * X(0) + X(0) * adag(0) * adag(1);

  std::cout << "Mixed sum and products: " << std::endl;

  std::cout << H.toString() << std::endl;
}
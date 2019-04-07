#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer = qcor::getOptimizer(
      "nlopt", {{"nlopt-optimizer", "cobyla"}, {"nlopt-maxeval", 1000}});

  auto op = qcor::getObservable(
      "pauli", "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  int nq = op->nBits();
  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](std::vector<double> x) {
          hwe(x, {{"n-qubits",nq},{"layers",1}});
        },
        op, optimizer);
  });

  auto results = future.get();
  std::cout << "Results:\n";
//   results->print();
}

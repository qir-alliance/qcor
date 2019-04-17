#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer = qcor::getOptimizer(
      "nlopt", {{"nlopt-optimizer", "cobyla"}, {"nlopt-maxeval", 20}});

  auto op = qcor::getObservable(
      "pauli", "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](double x) {
          X(0);
          exp_i_theta(x, {{"pauli", "X0 Y1 - Y0 X1"}});
        },
        op, optimizer);
  });

  auto results = future.get();
  auto energy = mpark::get<double>(results->getInformation("opt-val"));
  std::cout << "Results: " << energy << "\n";
}

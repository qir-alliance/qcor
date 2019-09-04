#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 2000)});

  auto observable =
      qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + 9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2"));

  auto handle = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](qbit q, double t0, double t1) {
          X(q[0]);
          exp_i_theta(q, t0, {{"pauli", "X0 Y1 - Y0 X1"}});
          exp_i_theta(q, t1, {{"pauli", "X0 Z1 Y2 - X2 Z1 Y0"}});
        },
        observable, optimizer, 0.0, 0.0);
  });

  auto results = qcor::sync(handle);
  
  auto energy = mpark::get<double>(results->getInformation("opt-val"));
  std::cout << "Results: " << energy << "\n";
}

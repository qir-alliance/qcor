#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 2000)});

  auto observable =
      qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 "
                                               "- 2.1433 Y0Y1"
                                               "+ .21829 Z0 - 6.125 Z1"));
  auto handle = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](qbit q, double x) {
          X(q[0]);
          exp_i_theta(q, x, {{"pauli", "X0 Y1 - Y0 X1"}});
        },
        observable, optimizer, 0.0);
  });

  auto results = qcor::sync(handle);
  auto energy = results->getInformation("opt-val").as<double>();
  std::cout << "Results: " << energy << "\n";

}

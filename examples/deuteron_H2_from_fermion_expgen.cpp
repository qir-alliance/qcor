#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 100)});

  auto observable =
      qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 "
                                               "- 2.1433 Y0Y1"
                                               "+ .21829 Z0 - 6.125 Z1"));

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](qbit q, double x) {
          X(q[0]);
          exp_i_theta(q, x, {{"fermion", "0^ 1 - 1^ 0"}});
        },
        observable, optimizer, 0.0);
  });

  auto results = future.get();
  auto energy = results->getInformation("opt-val").as<double>();
  std::cout << "Results: " << energy << "\n";
}

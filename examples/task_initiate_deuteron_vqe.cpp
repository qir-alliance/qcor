#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto ansatz = [&](qbit q, std::vector<double> t) {
    X(q[0]);
    Ry(q[1], t[0]);
    CNOT(q[1], q[0]);
  };

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 20)});

  auto observable =
      qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 "
                                               "- 2.1433 Y0Y1"
                                               "+ .21829 Z0 - 6.125 Z1"));

  auto handle = qcor::taskInitiate(ansatz, "vqe", optimizer, observable, std::vector<double>{0.0});
  auto results = qcor::sync(handle);

  std::cout << results->getInformation("opt-val").as<double>() << "\n";

  qcor::Finalize();
}

#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto ansatz = [&](qbit q, std::vector<double> t) {
    X(q[0]);
    X(q[1]);
    exp_i_theta(q, t, {{"pauli", "Y0 X1 X2 X3"}});
  };

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 200)});

  auto handle = qcor::taskInitiate(ansatz, "vqe", optimizer, std::vector<double>{0.0});
  auto results = qcor::sync(handle);

  std::cout << results->getInformation("opt-val").as<double>() << "\n";

  qcor::Finalize();
}

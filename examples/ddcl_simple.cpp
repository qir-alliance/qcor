#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto ansatz = [&](qbit q, std::vector<double> x) {
    Rx(q[0], x[0]);
    Ry(q[0], x[1]);
    Rx(q[0], x[2]);
  };

  auto optimizer =
      qcor::getOptimizer("nlopt", std::make_pair("nlopt-maxeval", 20));

  HeterogeneousMap ddclOptions{
      std::make_pair("loss", "js"),
      std::make_pair("target_dist", std::vector<double>{.5, .5})};

  auto handle = qcor::taskInitiate(ansatz, "ddcl", optimizer, ddclOptions,
                                   std::vector<double>{});
  auto results = qcor::sync(handle);

  std::cout << "JS: " << results->getInformation("opt-val").as<double>()
            << "\n";
}

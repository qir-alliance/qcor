#include "qcor.hpp"
#include <heterogeneous.hpp>

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto ansatz = [&](qbit q, std::vector<double> x) {
    hwe(q, x, {{"nq",2}, {"layers",1}});
  };

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 2000)});

  auto observable =
      qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 "
                                               "- 2.1433 Y0Y1"
                                               "+ .21829 Z0 - 6.125 Z1"));
  int nq = observable->nBits();

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(ansatz,
        observable, optimizer, std::vector<double>{});
  });

  auto results = future.get();
}

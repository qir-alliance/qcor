#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer = qcor::getOptimizer(
      "nlopt", {{"nlopt-optimizer", "cobyla"}, {"nlopt-maxeval", 1000}});

  auto geom = R"geom(2

H          0.00000        0.00000        0.00000
H          0.00000        0.00000        0.7474)geom";

  auto op = qcor::getObservable("chemistry", {{"basis","sto-3g"}, {"geometry", geom}});
  int nq = op->nBits();

  std::vector<std::pair<int,int>> coupling{{0,1},{1,2},{2,3}};

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](std::vector<double> x) {
          hwe(x, {{"n-qubits",nq},{"layers",1}, {"coupling",coupling}});
        },
        op, optimizer);
  });

  auto results = future.get();
  std::cout << "Results:\n";
//   results->print();
}

#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer = qcor::getOptimizer(
      "nlopt", {{"nlopt-optimizer", "cobyla"}, {"nlopt-maxeval", 1000}});

  auto geom = R"geom(2

H          0.00000        0.00000        0.00000
H          0.00000        0.00000        0.7474)geom";

  auto op = qcor::getObservable("chemistry",
                                {std::make_pair("basis", "sto-3g"), std::make_pair("geometry", geom)});
  int nq = op->nBits();

  std::vector<std::pair<int, int>> coupling{{0, 1}, {1, 2}, {2, 3}};

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](qbit q, std::vector<double> x) {
          hwe(q, x, {{"n-qubits", nq}, {"layers", 1}, {"coupling", coupling}});
        },
        op, optimizer);
  });

  auto results = future.get();
  auto energy = mpark::get<double>(results->getInformation("opt-val"));
  std::cout << "Results: " << energy << "\n";
  //   results->print();
}

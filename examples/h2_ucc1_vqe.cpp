#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer = qcor::getOptimizer(
      "nlopt", {{"nlopt-optimizer", "cobyla"}, {"nlopt-maxeval", 1000}});

  auto geom = R"geom(2

H          0.00000        0.00000        0.00000
H          0.00000        0.00000        0.7474)geom";

  auto op = qcor::getObservable("chemistry",
                                {{"basis", "sto-3g"}, {"geometry", geom}});
  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](double x) {
          X(0);
          X(2);
          Rx(1.57,0);
          H(1);
          H(2);
          H(3);
          CX(0,1);
          CX(1,2);
          CX(2,3);
          Rz(x, 3);
          CX(2,3);
          CX(1,2);
          CX(0,1);
          Rx(-1.57,0);
          H(1);
          H(2);
          H(3);
        },
        op, optimizer);
  });

  auto results = future.get();
  auto energy = mpark::get<double>(results->getInformation("opt-val"));
  std::cout << "Results: " << energy << "\n";
  //   results->print();
}

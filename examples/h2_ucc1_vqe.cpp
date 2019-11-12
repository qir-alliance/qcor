#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 2000)});

  auto geom = R"geom(2

H          0.00000        0.00000        0.00000
H          0.00000        0.00000        0.7474)geom";

  auto op = qcor::getObservable("chemistry",
                                {std::make_pair("basis", "sto-3g"), 
                                 std::make_pair("geometry", geom)});

  auto handle = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](qbit q, double x) {
          X(q[0]);
          X(q[2]);
          Rx(q[0], 1.57);
          H(q[1]);
          H(q[2]);
          H(q[3]);
          CX(q[0],q[1]);
          CX(q[1],q[2]);
          CX(q[2],q[3]);
          Rz(q[3], x);
          CX(q[2],q[3]);
          CX(q[1],q[2]);
          CX(q[0],q[1]);
          Rx(q[0], -1.57);
          H(q[1]);
          H(q[2]);
          H(q[3]);
        },
        op, optimizer, 0.0);
  });

  auto results = qcor::sync(handle);
  auto energy = results->getInformation("opt-val").as<double>();
  std::cout << "Results: " << energy << "\n";
}

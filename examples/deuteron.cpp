#include "qcor.hpp"

int main() {

  qcor::Initialize({"--accelerator", "tnqvm"});

  auto optimizer = qcor::getOptimizer("nlopt");

  const std::string deuteronH2 =
      R"deuteronH2((5.907,0) + (-2.1433,0) X0 X1 + (-2.1433,0) Y0 Y1 + (.21829,0) Z0 + (-6.125,0) Z1)deuteronH2";
  PauliOperator op;
  op.fromString(deuteronH2);

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](double t0) {
          X(0);
          Ry(t0, 1);
          CX(1, 0);
        },
        op, optimizer);
  });

  auto results = future.get();
  results->print();
}


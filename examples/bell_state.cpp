#include "qcor.hpp"

int main(int argc, char** argv) {

  qcor::Initialize(argc, argv);

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.execute([&](qbit q) {
      H(q[0]);
      CX(q[0], q[1]);
      Measure(q[0]);
      Measure(q[1]);
    });
  });

  // You just launched an async call,
  // go do other work if necessary...

  auto results = future.get();
  results->print();
}

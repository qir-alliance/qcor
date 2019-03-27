#include "qcor.hpp"

int main() {

  qcor::Initialize({"--accelerator", "local-ibm"});

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.execute([&]() { 
      H(0);
      CX(0, 1);
      Measure(0);
      Measure(1);
    });
  });

  // You just launched an async call,
  // go do other work if necessary...

  auto results = future.get();
  results->print();
}
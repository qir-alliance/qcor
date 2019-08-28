#include "qcor.hpp"

int main(int argc, char** argv) {

  qcor::Initialize(argc, argv);

  auto ansatz = [&](qbit q, std::vector<double> t) {
      X(q[0]);
      Ry(q[1], t[0]);
      CNOT(q[1], q[0]);
  };

//   qbit buffer = xacc::qalloc(2);
//   ansatz(buffer, std::vector<double>{2.2});
//   buffer->print();


  auto optimizer = qcor::getOptimizer(
      "nlopt", {std::make_pair("nlopt-optimizer","cobyla"),
                std::make_pair("nlopt-maxeval", 20)});

  auto observable = qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 "
                                         "- 2.1433 Y0Y1"
                                         "+ .21829 Z0 - 6.125 Z1"));

  // Schedule an asynchronous VQE execution
  // with the given quantum kernel ansatz
  std::vector<double> initial_params{0.0};
  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(ansatz, observable, optimizer, std::vector<double>{0.0});
  });
  auto results = future.get();
  std::cout << results->getInformation("opt-val").as<double>() << "\n";
}

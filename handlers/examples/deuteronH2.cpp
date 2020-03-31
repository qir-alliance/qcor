#include "qcor.hpp"
#include <vector> 

__qpu__ void ansatz(qreg q, double t) {
  X(q[0]);
  Ry(q[1], t);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {

  auto opt = qcor::getOptimizer();
  auto obs = qcor::getObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Schedule an asynchronous VQE execution
  // with the given quantum kernel ansatz
  auto handle = qcor::taskInitiate(ansatz, "vqe", opt, obs, 0.45);

  auto results_buffer = handle.get();
  auto energy = qcor::extract_results<double>(results_buffer, "opt-val");
  auto angles =
      qcor::extract_results<std::vector<double>>(results_buffer, "opt-params");

  printf("energy = %f\n", energy);
  printf("angles = [");
  for (int i = 0; i < 1; i++)
    printf("%f ", angles[i]);
  printf("]\n");
}

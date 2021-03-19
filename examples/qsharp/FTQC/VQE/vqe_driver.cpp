#include <iostream> 
#include <vector>
#include "qir-types.hpp"

// Include the external QSharp function.
qcor_include_qsharp(QCOR__DeuteronVqe__body, double, int64_t shots, Callable* opt_stepper);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qrt ftqc vqe_ansatz.qs vqe_driver.cpp
// Run with:
// $ ./a.out
int main() {
  std::function<std::vector<double>(double, std::vector<double>)> stepper =
      [&](double in_costVal, std::vector<double> previous_params) -> std::vector<double> {
        std::cout << "HELLO CALLBACK!\n";
        return {1.0};
      };
  qcor::qsharp::CallBack<std::vector<double>, double, std::vector<double>> cbFunc(
      stepper);

  Callable cb(&cbFunc);
  const double exp_val_xx = QCOR__DeuteronVqe__body(1024, &cb);
  return 0;
}
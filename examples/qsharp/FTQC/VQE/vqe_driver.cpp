#include <iostream> 
#include <vector>
#include "opt_stepper.hpp"
#include "qir-types-utils.hpp"

// Include the external QSharp function.
qcor_include_qsharp(QCOR__DeuteronVqe__body, double, int64_t shots, Callable* opt_stepper);


// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qrt ftqc vqe_ansatz.qs vqe_driver.cpp
// Run with:
// $ ./a.out
int main() {
  // Create an optimizer:
  qcor::OptStepper qcorOptimizer("nlopt", {{"maxeval", 20}});
  using StepperFuncType =
      std::function<std::vector<double>(double, std::vector<double>)>;

  // Create an optimizer stepper as a lambda function to provide to Q#.
  StepperFuncType stepper_callable = [&](double in_costVal,
                                         std::vector<double> previous_params) {
    static size_t iterCount = 0;
    // Update the stepper with new data
    qcorOptimizer.update(previous_params, in_costVal);
    std::cout << "Iter " << iterCount++ << ": Energy(" << previous_params[0]
              << ") = " << in_costVal << "\n";
    // Returns a new set of params for Q# to try.
    return qcorOptimizer.getNextParams();
  };

  // Run the Q# Deuteron with the *nlopt* stepper provided as a callable.
  // Note: qsharp::createCallable will marshal the C++ function (lambda) to the
  // Q# Callable type.
  const int64_t nb_shots = 2048;
  const double final_energy = QCOR__DeuteronVqe__body(
      nb_shots, qir::createCallable(stepper_callable));
  std::cout << "Final energy = " << final_energy << "\n";
  return 0;
}
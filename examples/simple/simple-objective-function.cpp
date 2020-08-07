// Note no includes here, we are just 
// using the language extension
//
// run this with 
// qcor -qpu tnqvm simple-objective-function.cpp
// ./a.out

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian (Observable)
  auto H = createObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  auto objective = createObjectiveFunction("vqe", ansatz, H);

  // Create the Optimizer. This will give us COBYLA from nlopt
  auto optimizer = createOptimizer("nlopt");

  // Launch the Optimization Task with taskInitiate
  auto handle = taskInitiate(
      objective, optimizer, // Provide the ObjectiveFunction and Optimizer
      // Need to provide a way to map vector<double> x to ansatz(qreg, double)
      // QCOR provides the TranslationFunctor for this
      TranslationFunctor<qreg, double>([&](const std::vector<double> x) {
        return std::make_tuple(q, x[0]);
      }),
      // Need to specify number of parameters in x
      // because qcor has to instantiate that vector
      1);

  // Go do other work...

  // Query results when ready.
  auto results = qcor::sync(handle);
  printf("vqe-energy from taskInitiate = %f\n", results.opt_val);

  // Evaluate the ObjectiveFunction at a specified set of parameters
  auto energy = (*objective)(q, results.opt_params[0]);
  printf("vqe-energy just evaluating objective function = %f\n", energy);
}

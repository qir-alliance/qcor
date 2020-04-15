#include "qcor.hpp"

// QCOR kernel requirements:
// C-like function, unique function name
// takes qreg as first argument, can take any 
// arguments after that which are necessary for function 
// evaluation (like gate rotation parameters). 
// Function body is written in a 'supported language' (i.e. 
// we have a xacc::Compiler parser for it, here XASM (which is default))
// Must be annotated with the __qpu__ attribute, which expands 
// to [[clang::syntax(qcor)]], thereby invoking our custom Clang SyntaxHandler. 

// Demonstrate a couple ways to program this 
// circuit ansatz, first with a double parameter
__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1],q[0]);
}

// Second with a vector parameter
__qpu__ void ansatz2(qreg q, std::vector<double> theta) {
  X(q[0]);
  Ry(q[1], theta[0]);
  CX(q[1],q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian (Observable)
  auto H = qcor::getObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  // Must also provide initial params for ansatz (under the hood, uses 
  // variadic template)
  auto objective = qcor::createObjectiveFunction("vqe", ansatz, H, q, 0.0);

  // Evaluate the ObjectiveFunction at a specified set of parameters
  auto energy = (*objective)(q, .59);

  // Print the result
  printf("vqe energy = %f\n", energy);
  q.print();

  // Clear and lets try to do the 
  // same thing with the vector anstaz
  q.reset();

  // Create the Objective again
  auto objective2 = qcor::createObjectiveFunction("vqe", ansatz2, H, q, std::vector<double>{0.0});

  // Evaluate, but with a vector
  energy = (*objective2)(q, std::vector<double>{.59});

  // Print the result
  printf("vqe energy = %f\n", energy);

  return 0;
}

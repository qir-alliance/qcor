// Run a suite of circuit optimization benchmarks.
#include "qcor.hpp"
#include <vector>
#include <string>
// Run custom passes
#ifdef TEST_SOURCE_FILE
__qpu__ void testKernel(qreg quVar) {
  using qcor::openqasm;
#include TEST_SOURCE_FILE
}

// Pass the ordered list of passes to run
// Note: should be at run at opt-level 0 since we 
// customize the list of passes as CLI arguments
int main(int argc, char *argv[]) {
  // Allocate just 1 qubit, we don't actually want to run the simulation.
  auto q = qalloc(1);
  {
    class testKernel t(q);
    t.optimize_only = true;
  }
  for (int i = 1; i < argc; ++i) {
    xacc::internal_compiler::execute_pass(argv[i]);
  }

  std::cout << "NInsts: " << quantum::program->nInstructions() << "\n";
}
#endif
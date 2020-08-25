// To use the QCOR JIT utilities 
// just include the qcor_jit.hpp header
#include "qcor_jit.hpp"

int main() {

  // QJIT is the entry point to QCOR quantum kernel 
  // just in time compilation
  QJIT qjit;

  // Define a quantum kernel string dynamically
  const auto kernel_src = R"#(__qpu__ void bell(qreg q) {
        using qcor::openqasm;
        h q[0];
        cx q[0], q[1];
        creg c[2];
        measure q -> c;
    })#";

  // Use the QJIT instance to compile this at runtime
  qjit.jit_compile(kernel_src);

  // Now, one can get the compiled kernel as a 
  // functor to execute, must provide the kernel 
  // argument types as template parameters
  auto bell_functor = qjit.get_kernel<qreg>("bell");

  // Allocate some qubits and run the kernel functor
  auto q = qalloc(2);
  bell_functor(q);
  q.print();

  // Or, one can call the QJIT invoke method 
  // with the name of the kernel function and 
  // the necessary function arguments.
  auto r = qalloc(2);
  qjit.invoke("bell", r);
  r.print();

  // Note, if QCOR QJIT has not seen this kernel 
  // source code before, it will run through the 
  // entire JIT compile process. If you have run 
  // this JIT compile before, QCOR QJIT will read a 
  // cached representation of the kernel and load that, 
  // increasing JIT compile performance. 
}
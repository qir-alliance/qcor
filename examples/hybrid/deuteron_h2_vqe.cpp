// This demonstrates the high-level qcor-hybrid
// library and its usage for the variational
// quantum eigensolver. Specifically, it demonstrates
// the deuteron N=2 experiment and does so
// with 2 quantum kernels of differing input argument structure.
// Run with
// qcor -qpu DESIRED_BACKEND deuteron_h2_vqe.cpp
// ./a.out
// (DESIRED_BACKEND) could be something like qpp, aer, ibm, qcs, etc.

// start off, include the hybrid header
// this gives us the VQE class
#include "qcor_hybrid.hpp"

// Define one quantum kernel that takes
// double angle parameter
__qpu__ void ansatz(qreg q, double x) {
  X(q[0]);
  Ry(q[1], x);
  CX(q[1], q[0]);
}

// Define another quantum kernel that takes
// a vector double argument, we just use one of them
// but we also demo more complicated ansatz construction
// using the exp_i_theta instruction
__qpu__ void ansatz_vec(qreg q, std::vector<double> x) {
  X(q[0]);
  auto ansatz_exponent = qcor::X(0) * qcor::Y(1) - qcor::Y(0) * qcor::X(1);
  exp_i_theta(q, x[0], ansatz_exponent);
}

__qpu__ void xasm_open_qasm_mixed_ansatz(qreg q, double xx) {
  using qcor::openqasm;
  x q[0];
  using qcor::xasm;
  // openqasm doesn't handle parameterized
  // gates, so switching to xasm here
  Ry(q[1], xx);
  using qcor::openqasm;
  cx q[1], q[0];
}

int main() {

  // Define the Hamiltonian using the QCOR API
  auto H = 5.907 - 2.1433 * qcor::X(0) * qcor::X(1) -
           2.1433 * qcor::Y(0) * qcor::Y(1) + .21829 * qcor::Z(0) -
           6.125 * qcor::Z(1);

  // Create a VQE instance, must give it
  // the parameterized ansatz functor, the
  // Observable, and the total number of parameters
  qcor::VQE vqe(ansatz, H, 1);

  // Execute synchronously, indicating the argument structure after
  // the required qreg first argument
  const auto [energy, params] = vqe.execute<double>();
  std::cout << "<H>(" << params[0] << ") = " << energy << "\n";

  // Now do the same for the vector double ansatz, but
  // also demonstrate the async interface
  qcor::VQE vqe_vec(ansatz_vec, H, 1);
  auto handle = vqe_vec.execute_async<std::vector<double>>();

  // Can go do other work, quantum execution is happening on
  // separate thread

  // Get the energy, this call will kick off a wait if needed
  const auto [energy_vec, params_vec] = vqe_vec.sync(handle);

  std::cout << "<H>(" << params_vec[0] << ") = " << energy_vec << "\n";

  qcor::VQE vqe_openqasm(xasm_open_qasm_mixed_ansatz, H, 1);
  const auto [energy_oq, params_oq] = vqe_openqasm.execute<double>();

  std::cout << "<H>(" << params_oq[0] << ") = " << energy_oq << "\n";
}

__qpu__ void ansatz(qreg q, double x) {
  X(q[0]);
  Ry(q[1], x);
  CX(q[1], q[0]);
}

int main() {
  auto q = qalloc(2);

  // Map the ansatz to a unitary matrix
  UnitaryMatrix u_mat = ansatz::as_unitary_matrix(q, .59);

  // Create the Hamiltonian, map to a DenseMatrix (complex values)
  auto H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1) + 5.907;
  DenseMatrix Hmat = get_dense_matrix(H);

  // Create the initial |0> state
  DenseVector init_state = DenseVector::Zero(4);
  init_state(0) = 1.;

  // Compute |psi> = U |0>
  DenseVector final_state = u_mat * init_state;

  // Compute the energy E = <psi | H | psi>
  std::complex<double> energy = final_state.transpose() * Hmat * final_state;
  std::cout << energy.real() << "\n";
}

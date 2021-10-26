__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main() {
  // Create the Hamiltonian
  auto H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1) + 5.907;

  const double test_theta = 1.2345;
  auto q = qalloc(2);
  const double energy = ansatz::observe(H, q, test_theta);
  std::cout << "E(" << test_theta << ") = " << energy << "\n";
  q.print();
}
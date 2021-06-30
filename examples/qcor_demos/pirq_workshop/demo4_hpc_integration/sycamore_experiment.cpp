// Fermionic Simulation gate:
// FSimGate(θ, φ) = 
// [[1, 0, 0, 0],
//  [0, a, b, 0],
//  [0, b, a, 0],
//  [0, 0, 0, c]]
// where:
// a = cos(theta)
// b = -i·sin(theta)
// c = exp(-i·phi)
__qpu__ void FSimGate(qubit q0, qubit q1, double theta, double phi) {
  std::vector<qubit> qubits{q0, q1};
  qreg q(qubits);
  auto a = std::cos(theta);
  auto b = std::complex<double>{ 0.0, -std::sin(theta)};
  auto c = std::exp(std::complex<double>{ 0.0, -phi});
  // Use QCOR unitaty decomposition (KAK)
  decompose {
    // Create the unitary matrix
    UnitaryMatrix fsim_mat = UnitaryMatrix::Identity(4, 4);
    fsim_mat(1, 1) = a;
    fsim_mat(1, 2) = b;
    fsim_mat(2, 1) = b;
    fsim_mat(2, 2) = a;
    fsim_mat(3, 3) = c;
  }(q, kak);
}


// TODO: Sycamore experiment...
__qpu__ void sycamore(qreg q) {
  FSimGate(q[1], q[4], 1.5157741664070026, 0.5567125777724111);
  FSimGate(q[3], q[7], 1.5177580142210796, 0.49481085782254924);
}

int main() {
  auto q = qalloc(53);
  sycamore::print_kernel(q);
}



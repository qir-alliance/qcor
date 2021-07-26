/// iqpe_ftqc.cpp: Iterative quantum phase estimation
/// Compile:
/// $ qcor -qrt ftqc iqpe_ftqc.cpp
/// Execute: 
/// ./a.out

// To show XACC IR execution (FTQC): add -print-final-submission option


// Iterative Quantum Phase Estimation:
// Only use 2 qubits to achieve 4-bit accuracy (normally require 5 qubits)
// The oracle is a -5*pi/8 phase rotation;
__qpu__ void oracle(qubit q) {
  auto angle = -5.0 * M_PI / 8.0;
  U1(q, angle);
}

using OracleCallable = KernelSignature<qubit>;
__qpu__ void phase_estimation(qreg q, OracleCallable oracle_fn) {
  X(q[1]);
  
  // Iterative phase estimation:
  H(q[0]);
  // C-U^8
  for (int i = 0; i < 8; i++) {
    oracle_fn.ctrl(q[0], q[1]);
  }
  H(q[0]);
  // Measure and reset
  // First bit:
  auto c0 = Measure(q[0]);
  Reset(q[0]);
  
  
  H(q[0]);
  // C-U^4
  for (int i = 0; i < 4; i++) {
    oracle_fn.ctrl(q[0], q[1]);
  }
  // Conditional phase correction based on previous results:
  if(c0) {
    Rz(q[0], -M_PI/2.0);
  }
  H(q[0]);
  // Second bit
  auto c1 = Measure(q[0]);
  Reset(q[0]);
  

  H(q[0]);
  // C-U^2
  for (int i = 0; i < 2; i++) {
    oracle_fn.ctrl(q[0], q[1]);
  }
  // Conditional phase correction based on previous results:
  if(c0) {
    Rz(q[0], -M_PI/4);
  }
  if(c1) {
    Rz(q[0], -M_PI/2);
  }
  H(q[0]);
  // 3rd bit:
  auto c2 = Measure(q[0]);
  Reset(q[0]);
  
  H(q[0]);
  // C-U^1
  oracle_fn.ctrl(q[0], q[1]);
  // Conditional phase correction based on previous results:
  if(c0) {
    Rz(q[0], -M_PI/8);
  }
  if(c1) {
    Rz(q[0], -M_PI/4);
  }
  if(c2) {
    Rz(q[0], -M_PI/2);
  }
  H(q[0]);
  // 4th bit:
  auto c3 = Measure(q[0]);
  
  // Print the result bit-string:
  print("Bit-string: ", c3, c2, c1, c0);
}

int main(int argc, char *argv[]) {
  qreg q = qalloc(2);
  // expected to get 4 bits (iteratively) of 1011 = 11(decimal):
  // phi_est = 11/16 (denom = 16 since we have 4 bits)
  // => phi = 2pi * 11/16 = 11pi/8 = 2pi - 5pi/8
  // i.e. we estimate the -5*pi/8 angle...
  phase_estimation(q, oracle);
  return 0;
}
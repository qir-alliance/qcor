#include <qcor_arithmetic>

__qpu__ void test_mul_integer_inline(qreg x, int a, int N, int &result) {
  // x = |1> + |3>
  X(x[0]);
  H(x[1]);
  // ==> |a*x> inline (save in x)
  // Note that this kernel will allocate qubits...
  mul_integer_mod_in_place(x, a, N);
  result = 0;
  for (int bitIdx = 0; bitIdx < x.size(); ++bitIdx) {
    if (Measure(x[bitIdx])) {
      result = result + (1 << bitIdx);
      X(x[bitIdx]);
    }
  }
}

int main(int argc, char **argv) {
  // Test in-place modular multiplication:
  // |x> ==> |ax mod N> on the same register.
  // x = |1> + |3>; a = 2
  // --> |2> + |6>
  // Simple test:
  int a_val = 2;
  int N_val = 8;
  auto x_reg2 = qalloc(3);
  int result = 0;
  test_mul_integer_inline(x_reg2, a_val, N_val, result);
  std::cout << "Result = " << result << "\n";
  qcor_expect(result == 2 || result == 6);
  return 0;
}
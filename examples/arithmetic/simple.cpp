#include <qcor_arithmetic>

__qpu__ void test_adder(qreg a, qreg b, qubit cin, qubit cout) {
  X(a[0]); // Set input a = 01
  X(b);    // Set input b = 11
  // Apply the adder
  ripple_add(a, b, cin, cout);
  Measure(b);
  Measure(cout);
}

// Compile:
int main(int argc, char **argv) {
  // Set the inputs to the adder
  auto a = qalloc(2);
  auto b = qalloc(2);
  auto carry_in = qalloc(1);
  auto carry_out = qalloc(1);
  // Execute:
  // test_adder();
  // test_adder::print_kernel(a, b, carry_in[0], carry_out[0]);
  test_adder(a, b, carry_in[0], carry_out[0]);
  b.print(); // 00
  carry_out.print(); // 1

  return 0;
}
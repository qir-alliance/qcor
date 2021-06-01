// Generate a random bit string in an obscure way...
__qpu__ void test_random(qreg q) {
  for (int i = 0; i < q.size(); ++i) {
    // Allocate qubit:
    // This qubit should be reused...
    auto anc_reg = qalloc(1);
    H(anc_reg);
    // entangle each qubit in the register
    // -> reset anc => q[i] in random 0 or 1 state
    X::ctrl(anc_reg[0], q[i]);
    Reset(anc_reg[0]);
  }
  Measure(q);
}

int main(int argc, char **argv) {
  set_shots(1024);
  auto a = qalloc(8);
  test_random(a);
  a.print();
  return 0;
}
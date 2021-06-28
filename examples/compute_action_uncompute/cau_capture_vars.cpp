__qpu__ void test(qreg q, double x) {
  std::vector<int> b{10, 20, 30};
  int j = 10;
  compute {
    int i = j;
    std::vector<int> bits = b;
    j = 22;

    print(i);
    print(bits[0], bits[1], bits[2]);
    H(q[0]);
    X::ctrl(q[0], q[1]);
  }
  action { Rz(q[1], x); }
}

int main() {
  auto q = qalloc(2);
  test::print_kernel(q, 2.2);
}

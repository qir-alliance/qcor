#include <qalloc>

__qpu__ void test_xasm(qreg q) {
  using qcor::xasm;
  H(q[0]);
  CX(q[0], q[1]);
  for (int i = 0; i < 2; i++)
    X(q[i]);
  Measure(q[0]);
  Measure(q[1]);
}

__qpu__ void test_openqasm(qreg q) {
    using qcor::openqasm;

    creg result[2];

    h q[0];
    cx q[0], q[1];

    measure q[0] -> result[0];
    measure q[1] -> result[1];
}


int main() {
  auto q = qalloc(2);
  test_xasm(q);
  q.print();

  auto qq = qalloc(2);
  test_openqasm(qq);
  qq.print();

}

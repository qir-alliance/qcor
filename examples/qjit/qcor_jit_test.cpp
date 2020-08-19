
#include "qcor.hpp"
#include "qcor_jit.hpp"
int main() {

  QJIT qjit;
  qjit.jit_compile(R"#(__qpu__ void bell(qreg q) {
        printf("hello world\n");
        using qcor::openqasm;
        h q[0];
        cx q[0], q[1];
        creg c[2];
        measure q -> c;
    })#");

  auto q = qalloc(2);
  qjit.invoke("bell", q);

  q.print();
}
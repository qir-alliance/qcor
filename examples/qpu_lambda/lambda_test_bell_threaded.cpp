#include "qcor.hpp"
// for _QCOR_MUTEX
#include "qcor_config.hpp"
#ifdef _QCOR_MUTEX
#include <mutex>
#include <thread>
#endif

void foo() {
  auto x_lambda = qpu_lambda([](qubit q) {
    X(q); });

  auto bell = qpu_lambda([](qreg q) {
    H(q[0]);
    // Call the captured lambda
    x_lambda.ctrl(q[0], q[1]);
    Measure(q);
  }, x_lambda);

  auto q = qalloc(2);
  bell(q);
  q.print();
  qcor_expect(q.counts().size() == 2);
  qcor_expect(q.counts()["00"] > 400);
  qcor_expect(q.counts()["11"] > 400);
  // Entangled...
  qcor_expect(q.counts()["00"] + q.counts()["11"] == 1024);
}

int main(int argc, char **argv) {
  set_shots(1024);
#ifdef _QCOR_MUTEX
  std::cout << "_QCOR_MUTEX is defined: multi-threding execution" << std::endl;
  std::thread t0(foo);
  std::thread t1(foo);
  t0.join();
  t1.join();
#else
  std::cout << "_QCOR_MUTEX is NOT defined: sequential execution" << std::endl;
  foo();
  foo();
#endif
}

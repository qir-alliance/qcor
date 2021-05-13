#include "qcor.hpp"
// for _XACC_MUTEX
#include "xacc.hpp"

void foo() {

  auto ansatz_X0X1 = qpu_lambda([](qreg q, double x) {
    X(q[0]);
    Ry(q[1], x);
    CX(q[1], q[0]);
    H(q);
    Measure(q);
  });

  OptFunction obj(
      [&](const std::vector<double> &x, std::vector<double> &) {
        auto q = qalloc(2);
        ansatz_X0X1(q, x[0]);
        auto exp = q.exp_val_z();
        print("<X0X1(",x[0],") = ", exp);
        return exp;
      },
      1);

  auto optimizer = createOptimizer(
      "nlopt",
      {{"initial-parameters", std::vector<double>{1.2}}, {"maxeval", 10}});
  auto [opt_val, opt_params] = optimizer->optimize(obj);
  print("opt_val = ", opt_val);
}

int main(int argc, char **argv) {
#ifdef _XACC_MUTEX
  std::cout << "_XACC_MUTEX is defined: multi-threding execution" << std::endl;
  std::thread t0(foo);
  std::thread t1(foo);
  t0.join();
  t1.join();
#else
  std::cout << "_XACC_MUTEX is NOT defined: sequential execution" << std::endl;
  foo();
  foo();
#endif
}

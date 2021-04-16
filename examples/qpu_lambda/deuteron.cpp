#include "qcor.hpp"

int main() {
  qcor::qpu_lambda<> ansatz_X0X1(
      [](qreg q, double x) {
        qpu_lambda_body({
          X(q[0]);
          Ry(q[1], x);
          CX(q[1], q[0]);
          H(q);
          Measure(q);
        })
      },
      qcor::qpu_lambda_variables({"q", "x"}, {}));

  qcor::OptFunction obj(
      [&](const std::vector<double> &x, std::vector<double> &) {
          print("running ", x[0]);
        auto q = qalloc(2);
        ansatz_X0X1(q, x[0]);
        auto exp = q.exp_val_z();
        print(x[0], exp);
        return exp;
      },
      1);

  auto optimizer = createOptimizer(
      "nlopt", {{"initial-parameters", std::vector<double>{1.2}}, {"maxeval", 10}});
  auto [results, opt] = optimizer->optimize(obj);

  print("r = ", results);
}
#include <qcor_hadamard_test>
using FO = FermionOperator;

int main() {

  std::vector<FermionOperator> exp_args{
      -4 * adag(0) * a(0) - 4 * adag(2) * a(2) +
          8 * adag(0) * a(0) * adag(2) * a(2),
      0.01 * adag(0) * a(1) + .01 * adag(3) * a(2) + .01 * adag(1) * a(0) +
          .01 * adag(2) * a(3)};
  std::vector<double> trot_params{1.0, 1, 3};

  auto sp = qpu_lambda([](qreg q) {
    X(q[0]);
    X(q[1]);
  });

  auto unitary = qpu_lambda(
      [&](qreg q) {
        auto dt = trot_params[0];
        auto num_trot_steps = trot_params[1];
        auto tot_trot_steps = trot_params[2];
        double delta_t = dt / num_trot_steps;
        for (auto j : range(tot_trot_steps)) {
          for (auto i : range(num_trot_steps)) {
            exp_i_theta(q, .5 * delta_t, exp_args[0]);
            exp_i_theta(q, delta_t, exp_args[1]);
            exp_i_theta(q, .5 * delta_t, exp_args[0]);
          }
        }
      },
      trot_params, exp_args);

  auto val = qcor::hadamard_test(sp, unitary, 4);
  print(val);
}
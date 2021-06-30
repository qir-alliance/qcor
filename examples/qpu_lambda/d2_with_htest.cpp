#include <qcor_hadamard_test>

int main() {
  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  for (auto x : linspace(-constants::pi, constants::pi, 10)) {
    auto terms = H.getNonIdentitySubTerms();
    double sum = H.getIdentitySubTerm()->coefficient().real();
    for (auto term : terms) {
      auto pop = std::dynamic_pointer_cast<PauliOperator>(term);

      assert(pop && pop->nTerms() == 1);

      auto [zv, xv] = pop->toBinaryVectors(2);

      auto sp = qpu_lambda(
          [](qreg q) {
            X(q.head());
            Ry(q.tail(), x);
            X::ctrl(q.tail(), q.head());
          },
          x);

      auto l = qpu_lambda(
          [](qreg q) {
            for (auto [i, x_val] : enumerate(xv)) {
              auto z_val = zv[i];
              if (x_val == z_val && x_val == 1) {
                Y(q[i]);
              } else if (x_val == 1) {
                X(q[i]);
              } else if (z_val == 1) {
                Z(q[i]);
              }
            }
          },
          xv, zv);

      auto val = qcor::hadamard_test(sp, l, 2);
      sum += pop->coefficient().real() * val;
    }

    print("E(", x, ") = ", sum);
  }
}
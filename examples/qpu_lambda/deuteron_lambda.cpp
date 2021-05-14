int main() {
  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  auto terms = H.getNonIdentitySubTerms();
  // Test repeated lambda creation.
  for (auto term : terms) {
    auto pop = std::dynamic_pointer_cast<PauliOperator>(term);

    assert(pop && pop->nTerms() == 1);

    auto [zv, xv] = pop->toBinaryVectors(2);

    auto l = qpu_lambda(
        [](qreg q) {
          for (auto [i, x_val] : enumerate(xv)) {
            auto z_val = zv[i];
            if (x_val == z_val) {
              Y(q[i]);
            } else if (x_val == 0) {
              Z(q[i]);
            } else {
              X(q[i]);
            }
          }
        },
        xv, zv);
  }
}
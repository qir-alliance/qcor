

__qpu__ void ccnot(qreg q) {
  X(q);

  decompose {
    UnitaryMatrix ccnot_mat = UnitaryMatrix::Identity(8, 8);
    ccnot_mat(6, 6) = 0.0;
    ccnot_mat(7, 7) = 0.0;
    ccnot_mat(6, 7) = 1.0;
    ccnot_mat(7, 6) = 1.0;
  }
  (q, qfactor);

  Measure(q);
}

int main() {
  auto q = qalloc(3);

  ccnot(q);

  q.print();
}
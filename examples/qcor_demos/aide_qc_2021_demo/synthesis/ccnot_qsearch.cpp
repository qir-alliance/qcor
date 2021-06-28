

__qpu__ void ccnot(qreg q, std::vector<int> initial_state) {
  // Construct initial state
  for (auto [i, element] : enumerate(initial_state)) {
    if (element) X(q[i]);
  }

  decompose {
    UnitaryMatrix ccnot_mat = UnitaryMatrix::Identity(8, 8);
    ccnot_mat(6, 6) = 0.0;
    ccnot_mat(7, 7) = 0.0;
    ccnot_mat(6, 7) = 1.0;
    ccnot_mat(7, 6) = 1.0;
  }
  (q, qsearch);

  Measure(q);
}

int main() {
  std::vector<std::vector<int>> truth_table;
  for (int i = 0; i < 8; ++i) {
    truth_table.push_back({i / 4 % 2, i / 2 % 2, i % 2});
  }

  for (auto row : truth_table) {
    auto q = qalloc(3);
    ccnot(q, row);
    auto results = q.counts().begin()->first;
    printf("Input %d%d%d -> Output %s\n", row[0], row[1], row[2], results.c_str());
  }
  
  return;
}
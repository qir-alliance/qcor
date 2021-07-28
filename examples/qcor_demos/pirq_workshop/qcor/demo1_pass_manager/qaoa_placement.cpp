/// qaoa_placement.cpp: Topology placement example
/// Using the default placement:
/// qcor -qpu aer:ibmq_guadalupe  qaoa_placement.cpp -print-final-submission
/// Change the placement:
/// qcor -qpu aer:ibmq_guadalupe -placement enfield qaoa_placement.cpp -print-final-submission
/// Can add -opt 1 to further combine single-qubit gates (mixer terms).
__qpu__ void qaoa_maxcut(qreg q, std::vector<double> gamma,
                  std::vector<double> beta,
                  std::vector<std::pair<int, int>> graph_edges) {
  auto nQubits = q.size();
  int p = gamma.size();

  // Start of in the uniform superposition
  H(q);

  // Loop over qaoa steps
  for (int step = 0; step < p; step++) {
    // Loop over graph edges
    for (int i = 0; i < graph_edges.size(); i++) {
      auto [node1, node2] = graph_edges[i];
      auto ham_operator =  Z(node1) * Z(node2);
      // trotterize
      exp_i_theta(q, gamma[step], ham_operator);
    }

    // Add the reference hamiltonian term (mixer)
    for (int i = 0; i < nQubits; i++) {
      auto ref_ham_term = X(i);
      exp_i_theta(q, beta[step], ref_ham_term);
    }
  }

  Measure(q);
}

int main(int argc, char **argv) {
  auto q = qalloc(5);
  // Ring graph
  std::vector<std::pair<int, int>> graph{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}};
  // One step
  int p = 1;
  std::vector<double> gammas{1.0};
  std::vector<double> betas{1.0};
  qaoa_maxcut::print_kernel(q, gammas, betas, graph);
  qaoa_maxcut(q, gammas, betas, graph);
}

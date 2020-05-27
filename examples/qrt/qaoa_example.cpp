#include "qcor.hpp"

__qpu__ void qaoa_ansatz(qreg q, int n_steps, std::vector<double> gamma,
                         std::vector<double> beta,
                         std::shared_ptr<qcor::Observable> cost_ham) {

  auto nQubits = q.size();
  int gamma_counter = 0;
  int beta_counter = 0;

  for (int i = 0; i < nQubits; i++) {
    H(q[0]);
  }

  auto cost_terms = cost_ham->getNonIdentitySubTerms();

  for (int step = 0; step < n_steps; step++) {

    for (int i = 0; i < cost_terms.size(); i++) {

      // for xasm we have to allocate the variables
      // cant just run exp_i_theta(q, gamma[gamma_counter], cost_terms[i]) yet :(
      auto cost_term = cost_terms[i];
      auto m_gamma = gamma[gamma_counter];

      exp_i_theta(q, m_gamma, cost_term);

      gamma_counter++;
    }

    for (int i = 0; i < nQubits; i++) {
      auto ref_ham_term = qcor::createObservable("X" + std::to_string(i));
      auto m_beta = beta[beta_counter];
      exp_i_theta(q, m_beta, ref_ham_term);
      beta_counter++;
    }
  }
}

int main(int argc, char **argv) {
  // Allocate 4 qubits
  auto q = qalloc(4);

  int nGamma = 7;
  int nBeta = 4;
  int nParamsPerStep = nGamma + nBeta;
  int nSteps = 4;
  int total_params = nSteps * nParamsPerStep;

  std::vector<double> initial_params(total_params);
  std::generate(initial_params.begin(), initial_params.end(), std::rand);

  auto cost_ham =
      qcor::createObservable("-5.0 - 0.5 Z0 - 1.0 Z2 + 0.5 Z3 + 1.0 Z0 Z1 + "
                             "2.0 Z0 Z2 + 0.5 Z1 Z2 + 2.5 Z2 Z3");

  auto objective = qcor::createObjectiveFunction("vqe", qaoa_ansatz, cost_ham);

  // Create the Optimizer
  auto optimizer = qcor::createOptimizer("nlopt", {std::make_pair(
                                                      "initial-parameters", initial_params)
                                                  });

  auto k = qcor::__internal__::kernel_as_composite_instruction(
      qaoa_ansatz, q, nSteps, std::vector<double>(nGamma),
      std::vector<double>(nBeta), cost_ham);

  //   Create mechanism for mapping Optimizer std::vector<double> parameters
  //   to the ObjectiveFunction variadic arguments of qreg and double
  auto args_translation =
      qcor::TranslationFunctor<qreg, int, std::vector<double>,
                               std::vector<double>,
                               std::shared_ptr<qcor::Observable>>(
          [&](const std::vector<double> x) {
            // split x into gamma and beta sets
            std::vector<double> gamma(x.begin(), x.begin() + nSteps * nGamma),
                beta(x.begin() + nSteps * nGamma,
                     x.begin() + nSteps * nGamma + nSteps * nBeta);
            return std::make_tuple(q, nSteps, gamma, beta, cost_ham);
          });

  qcor::set_verbose(true);
  // Call taskInitiate, kick off optimization of the give
  // functor dependent on the ObjectiveFunction, async call
  // Need to translate Optimizer std::vector<double> x params to Objective
  // Function evaluation args qreg, double.
  auto handle =
      qcor::taskInitiate(objective, optimizer, args_translation, total_params);

  // Go do other work...

  // Query results when ready.
  auto results = qcor::sync(handle);

  // Print the optimal value.
  printf("Min QUBO value = %f\n", results.opt_val);
}

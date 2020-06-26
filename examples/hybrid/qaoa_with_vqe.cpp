#include "qcor_hybrid.hpp"

__qpu__ void qaoa_ansatz(qreg q, int n_steps, std::vector<double> gamma,
                         std::vector<double> beta,
                         qcor::PauliOperator &cost_ham) {

  // Local Declarations
  auto nQubits = q.size();
  int gamma_counter = 0;
  int beta_counter = 0;

  // Start of in the uniform superposition
  for (int i = 0; i < nQubits; i++) {
    H(q[0]);
  }

  // Get all non-identity hamiltonian terms
  // for the following exp(H_i) trotterization
  auto cost_terms = cost_ham.getNonIdentitySubTerms();

  // Loop over qaoa steps
  for (int step = 0; step < n_steps; step++) {

    // Loop over cost hamiltonian terms
    for (int i = 0; i < cost_terms.size(); i++) {

      // for xasm we have to allocate the variables
      // cant just run exp_i_theta(q, gamma[gamma_counter], cost_terms[i]) yet
      // :(
      auto cost_term = cost_terms[i];
      auto m_gamma = gamma[gamma_counter];

      // trotterize
      exp_i_theta(q, m_gamma, cost_term);

      gamma_counter++;
    }

    // Add the reference hamiltonian term
    for (int i = 0; i < nQubits; i++) {
      auto ref_ham_term = qcor::X(i);
      auto m_beta = beta[beta_counter];
      exp_i_theta(q, m_beta, ref_ham_term);
      beta_counter++;
    }
  }
}

int main(int argc, char **argv) {

  // Specify the size of the problem
  int nGamma = 7;
  int nBeta = 4;
  int nParamsPerStep = nGamma + nBeta;
  int nSteps = 3;
  int total_params = nSteps * nParamsPerStep;

  // Generate a random initial parameter set
  auto initial_gamma = qcor::random_vector(0., 1., nSteps * nGamma);
  auto initial_beta = qcor::random_vector(0., 1., nSteps * nBeta);

  // Construct the cost hamiltonian
  auto cost_ham =
      -5.0 - 0.5 * (qcor::Z(0) - qcor::Z(3) - qcor::Z(1) * qcor::Z(2)) -
      qcor::Z(2) + 2 * qcor::Z(0) * qcor::Z(2) + 2.5 * qcor::Z(2) * qcor::Z(3);

  // VQE needs a way to translate std::vector<double> parameters
  // coming from the OptFunction being optimized into arguments
  // for the quantum kernel. We call this a TranslationFunctor.
  // Provide one here that maps vector<double> x to gamma and beta
  // on the quantum kernel, with captured variables for the other args.
  auto args_translation =
      qcor::TranslationFunctor<qreg, int, std::vector<double>,
                               std::vector<double>, qcor::PauliOperator>(
          [&](const std::vector<double> x) {
            // split x into gamma and beta sets
            std::vector<double> gamma(x.begin(), x.begin() + nSteps * nGamma),
                beta(x.begin() + nSteps * nGamma,
                     x.begin() + nSteps * nGamma + nSteps * nBeta);
            return std::make_tuple(qalloc(4), nSteps, gamma, beta, cost_ham);
          });

  qcor::set_verbose(true);

  auto optimizer = qcor::createOptimizer(
      "nlopt", {std::make_pair("nlopt-optimizer", "l-bfgs"),
                std::make_pair("nlopt-maxeval", 100)});
  qcor::VQE vqe(qaoa_ansatz, cost_ham, args_translation);
  auto [energy, params] =
      vqe.execute(optimizer, nSteps, initial_gamma, initial_beta, cost_ham);

  // Print the optimal value.
  printf("Min QUBO value = %f\n", energy);
}

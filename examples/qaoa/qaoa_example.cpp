#include "qcor.hpp"
#include <random>
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
  // Allocate 4 qubits
  auto q = qalloc(4);

  // Specify the size of the problem
  int nGamma = 7;
  int nBeta = 4;
  int nParamsPerStep = nGamma + nBeta;
  int nSteps = 3;
  int total_params = nSteps * nParamsPerStep;

  // Generate a random initial parameter set
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{0, 1};
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  std::vector<double> initial_params(total_params);
  std::generate(initial_params.begin(), initial_params.end(), gen);

  // Construct the cost hamiltonian
  auto cost_ham =
      -5.0 - 0.5 * (qcor::Z(0) - qcor::Z(3) - qcor::Z(1) * qcor::Z(2)) -
      qcor::Z(2) + 2 * qcor::Z(0) * qcor::Z(2) + 2.5 * qcor::Z(2) * qcor::Z(3);

  // Create the VQE ObjectiveFunction, giving it the
  // ansatz and Observable (cost hamiltonian)
  auto objective = qcor::createObjectiveFunction("vqe", qaoa_ansatz, cost_ham);

  // Create the classical Optimizer
  auto optimizer = qcor::createOptimizer(
      "nlopt", {std::make_pair("initial-parameters", initial_params),
                std::make_pair("nlopt-maxeval", 100)});

  // Create mechanism for mapping Optimizer std::vector<double> parameters
  // to the ObjectiveFunction variadic arguments corresponding to the above
  // quantum kernel (qreg, int, vec<double>, vec<double>, PauliOperator)
  auto args_translation =
      qcor::TranslationFunctor<qreg, int, std::vector<double>,
                               std::vector<double>, qcor::PauliOperator>(
          [&](const std::vector<double> x) {
            // split x into gamma and beta sets
            std::vector<double> gamma(x.begin(), x.begin() + nSteps * nGamma),
                beta(x.begin() + nSteps * nGamma,
                     x.begin() + nSteps * nGamma + nSteps * nBeta);
            return std::make_tuple(q, nSteps, gamma, beta, cost_ham);
          });
  qcor::set_verbose(true);
  // Launch the job asynchronously
  auto handle =
      qcor::taskInitiate(objective, optimizer, args_translation, total_params);

  // Go do other work... if you want

  // Query results when ready.
  auto results = qcor::sync(handle);

  // Print the optimal value.
  printf("Min QUBO value = %f\n", results.opt_val);

}

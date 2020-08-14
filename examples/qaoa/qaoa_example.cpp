#include <random>

__qpu__ void qaoa_ansatz(qreg q, int n_steps, std::vector<double> gamma,
                         std::vector<double> beta, std::string cost_ham_str) {

  // Local Declarations
  auto nQubits = q.size();
  int gamma_counter = 0;
  int beta_counter = 0;

  // Start of in the uniform superposition
  for (int i = 0; i < nQubits; i++) {
    H(q[0]);
  }

  auto cost_ham_ptr = createObservable(cost_ham_str);
  auto& cost_ham = *cost_ham_ptr.get();

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
      auto ref_ham_term = X(i);
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
  auto initial_params = random_vector(0., 1., total_params);

  // Construct the cost hamiltonian
  auto cost_ham = -5.0 - 0.5 * (Z(0) - Z(3) - Z(1) * Z(2)) - Z(2) +
                  2 * Z(0) * Z(2) + 2.5 * Z(2) * Z(3);

  // FIXME, currently with args translator and make_tuple we aren't able 
  // to directly pass Observable&, so here we just map to a string and 
  // read the string to an Observable in the kernel
  auto args_translator =
      std::make_shared<ArgsTranslator<qreg, int, std::vector<double>,
                                      std::vector<double>, std::string>>(
          [&](const std::vector<double> x) {
            // split x into gamma and beta sets
            std::vector<double> gamma(x.begin(), x.begin() + nSteps * nGamma),
                beta(x.begin() + nSteps * nGamma,
                     x.begin() + nSteps * nGamma + nSteps * nBeta);
            return std::make_tuple(q, nSteps, gamma, beta, cost_ham.toString());
          });

  // Create the VQE ObjectiveFunction
  auto objective = createObjectiveFunction(qaoa_ansatz, cost_ham,
                                           args_translator, q, total_params);

  // Create the classical Optimizer
  auto optimizer = createOptimizer(
      "nlopt", {std::make_pair("initial-parameters", initial_params),
                std::make_pair("nlopt-maxeval", 100)});

  
  set_verbose(true);

  // Launch the job asynchronously
  auto handle =
      taskInitiate(objective, optimizer);

  // Go do other work... if you want

  // Query results when ready.
  auto results = sync(handle);

  // Print the optimal value.
  printf("Min QUBO value = %f\n", results.opt_val);
}

#include "OperatorPool.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

using ObservablePtr = std::shared_ptr<Observable>;

// Initial State is HF
__qpu__ void initial_state(qreg q) {
  X(q[0]);
  X(q[2]);
}

// Adapt Ansatz, grows as the vector of input Observables grows
__qpu__ void adapt_ansatz(qreg q, std::vector<double> x,
                          std::vector<ObservablePtr> ops) {
  initial_state(q);
  for (auto [i, op] : enumerate(ops)) {
    exp_i_theta(q, x[i], op);
  }
}

int main() {

  // Define the Hamiltonian using the QCOR API
  auto H = 0.1202 * Z(0) * Z(1) + 0.168336 * Z(0) * Z(2) +
           0.1202 * Z(2) * Z(3) + 0.17028 * Z(2) + 0.17028 * Z(0) +
           0.165607 * Z(0) * Z(3) + 0.0454063 * Y(0) * Y(1) * X(2) * X(3) -
           0.106477 - 0.220041 * Z(3) + 0.174073 * Z(1) * Z(3) +
           0.0454063 * Y(0) * Y(1) * Y(2) * Y(3) - 0.220041 * Z(1) +
           0.165607 * Z(1) * Z(2) + 0.0454063 * X(0) * X(1) * Y(2) * Y(3) +
           0.0454063 * X(0) * X(1) * X(2) * X(3);

  // Use XACC to generate the operator pool Ai
  auto pool =
      xacc::getService<xacc::quantum::OperatorPool>("singlet-adapted-uccsd");
  pool->optionalParameters({{"n-electrons", 2}});
  auto ops = pool->generate(H.nBits());

  // Compute all [H, Ai] commutators
  std::vector<ObservablePtr> commutators(ops.size());
  std::generate(commutators.begin(), commutators.end(),
                [n = 0, &H, &ops]() mutable {
                  auto c = H.commutator(ops[n]);
                  n++;
                  return c;
                });

  // Local Declarations
  double energy = 0.0;
  std::vector<double> x;
  std::vector<ObservablePtr> all_ops;

  // Get the Optimizer
  auto optimizer = createOptimizer("nlopt");

  // Allocate a qreg
  auto q = qalloc(H.nBits());

  // Create an ArgsTranslator to translate
  // vec<double> x -> qreg, vec<double> vec<ObservablePtr
  auto args_translator = std::make_shared<ArgsTranslator<
      qreg, std::vector<double>, std::vector<std::shared_ptr<Observable>>>>(
      [&](const std::vector<double> &xx) {
        return std::make_tuple(q, xx, all_ops);
      });

  // start ADAPT loop
  for (int iter = 0; iter < 100; iter++) {
    double max_grad = 0.0, grad_norm = 0.0;
    int max_grad_idx = 0, counter = 0;

    // Compute the gradient vector dEi = <[H,Ai]>
    std::vector<double> grad_vec(commutators.size());
    std::generate(grad_vec.begin(), grad_vec.end(), [&]() {
      // Create an ObjectiveFunction to compute <[H,Ai]>
      auto obj = createObjectiveFunction(adapt_ansatz, commutators[counter],
                                         args_translator, q, x.size());
      counter++;
      // compute the grad element <[H,Ai]>
      auto grad = (*obj)(x);
      // set the gradient norm
      grad_norm += grad * grad;
      return grad;
    });

    // grad_vec is filled, find the index of the max element
    auto max_idx = std::distance(
        grad_vec.begin(), std::max_element(grad_vec.begin(), grad_vec.end()));

    // Grow the list of operators to build the ansatz
    all_ops.push_back(ops[max_idx]);

    // Check for convergence
    if (grad_norm < 1e-6) {
      std::cout << "Converged on gradient norm\n";
      break;
    }

    // Add a new parameter to the ansatz params
    x.push_back(0.0);

    // Create another ObjectiveFunction to
    // run the VQE min
    auto vqe =
        createObjectiveFunction(adapt_ansatz, H, args_translator, q, x.size());

    // Set the initial parameters
    optimizer->appendOption("initial-parameters", x);

    // Run task initiate and sync
    auto handle = taskInitiate(vqe, optimizer);
    auto results = sync(handle);

    // Get the energy
    energy = results.opt_val;

    // Get the current optimal params
    x = results.opt_params;

    std::cout << "Current Adapt Energy = " << energy << "\n";
  }

  std::cout << "Adapt computing energy = " << energy << "\n";
  std::cout << "Number of Adapt Parameters = " << x.size() << "\n";
  std::cout << "Adapt Optimal Parameters = [";
  for (auto xx : x) {
    std::cout << xx << " ";
  }
  std::cout << "]\n";

  // Run the kernel
  {
    class adapt_ansatz t(qalloc(H.nBits()), x, all_ops);
    t.optimize_only = true;
    // kernel executed upon destruction,
    // will only build up circuit and run pass manager
  }
  std::cout << "NInsts: " << quantum::program->nInstructions() << "\n";
  std::cout << "Number of Adapt Ansatz Gates: "
            << quantum::program->nInstructions() << "\n";

  return 0;
}
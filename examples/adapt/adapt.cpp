#include "OperatorPool.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

__qpu__ void initial_state(qreg q) {
  X(q[0]);
  X(q[2]);
}
__qpu__ void adapt_ansatz(qreg q, std::vector<double> x,
                          std::vector<std::shared_ptr<Observable>> ops) {
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

  // Use XACC to generate the operator pool
  auto pool =
      xacc::getService<xacc::quantum::OperatorPool>("singlet-adapted-uccsd");
  pool->optionalParameters({{"n-electrons", 2}});
  auto ops = pool->generate(H.nBits());

  std::vector<std::shared_ptr<Observable>> commutators(ops.size());
  std::generate(commutators.begin(), commutators.end(),
                [n = 0, &H, &ops]() mutable {
                  auto c = H.commutator(ops[n]);
                  n++;
                  return c;
                });

  double energy = 0.0;
  std::vector<double> x; // these are the variational parameters

  std::vector<std::shared_ptr<Observable>> all_ops;
  auto optimizer = createOptimizer("nlopt");

  // start ADAPT loop
  for (int iter = 0; iter < 100; iter++) {

    double max_grad = 0.0, grad_norm = 0.0;
    int max_grad_idx = 0;
    for (auto [idx, commutator] : enumerate(commutators)) {
      auto q = qalloc(H.nBits());
      auto args_translator = std::make_shared<ArgsTranslator<
          qreg, std::vector<double>, std::vector<std::shared_ptr<Observable>>>>(
          [&](const std::vector<double> &x) {
            return std::make_tuple(q, x, all_ops);
          });
      auto obj = createObjectiveFunction(adapt_ansatz, commutator,
                                         args_translator, q, x.size());
      auto grad = (*obj)(x);
      if (grad > max_grad) {
        max_grad = grad;
        max_grad_idx = idx;
      }
      grad_norm += grad * grad;
    }

    auto new_op = ops[max_grad_idx];

    all_ops.push_back(new_op);

    if (grad_norm < 1e-6) {
      std::cout << "Converged on gradient norm\n";
      break;
    }

    // add to the new parameter
    x.push_back(0.0);

    auto q = qalloc(H.nBits());

    auto args_translator = std::make_shared<ArgsTranslator<
        qreg, std::vector<double>, std::vector<std::shared_ptr<Observable>>>>(
        [&](const std::vector<double> &xx) {
          return std::make_tuple(q, xx, all_ops);
        });

    optimizer->appendOption("initial-parameters", x);
    auto vqe =
        createObjectiveFunction(adapt_ansatz, H, args_translator, q, x.size());
    auto handle = taskInitiate(vqe, optimizer);
    auto results = sync(handle);
    auto e = results.opt_val;
    x = results.opt_params;
    energy = e;
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
#include "qcor.hpp"

#include "Optimizer.hpp"

#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_quantum_gate_api.hpp"

#include "qalloc.hpp"

namespace qcor {
void set_verbose(bool verbose) {xacc::set_verbose(verbose);}

namespace __internal__ {
std::shared_ptr<ObjectiveFunction> get_objective(const char * type) {
    return xacc::getService<ObjectiveFunction>(type);
}
std::vector<std::shared_ptr<xacc::CompositeInstruction>>
observe(std::shared_ptr<xacc::Observable> obs,
        xacc::CompositeInstruction *program) {
  return obs->observe(xacc::as_shared_ptr(program));
}

double observe(xacc::CompositeInstruction* program,
             std::shared_ptr<xacc::Observable> obs,
             xacc::internal_compiler::qreg &q) {
  return [program, obs, &q]() {
    // Observe the program
    auto programs = __internal__::observe(obs, program);

    std::vector<xacc::CompositeInstruction *> ptrs;
    for (auto p : programs) {
      ptrs.push_back(p.get());
    }

    xacc::internal_compiler::execute(q.results(), ptrs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }();
}
} // namespace __internal__

xacc::Optimizer *getOptimizer() {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getOptimizer("nlopt").get();
}

std::shared_ptr<xacc::Observable> getObservable(const char *repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::quantum::getObservable("pauli", std::string(repr));
}


} // namespace qcor
#include "qcor.hpp"

#include "Optimizer.hpp"

#include "qalloc.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_quantum_gate_api.hpp"
#include "xacc_service.hpp"
#include <memory>

namespace qcor {
void set_verbose(bool verbose) { xacc::set_verbose(verbose); }
void set_shots(const int shots) { ::quantum::set_shots(shots); }

namespace __internal__ {
std::shared_ptr<ObjectiveFunction> get_objective(const std::string &type) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getService<ObjectiveFunction>(type);
}
std::shared_ptr<xacc::IRTransformation>
get_transformation(const std::string &transform_type) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getService<xacc::IRTransformation>(transform_type);
}

std::vector<std::shared_ptr<xacc::CompositeInstruction>>
observe(std::shared_ptr<xacc::Observable> obs,
        std::shared_ptr<CompositeInstruction> program) {
  return obs->observe(program);
}

double observe(std::shared_ptr<CompositeInstruction> program,
               std::shared_ptr<xacc::Observable> obs,
               xacc::internal_compiler::qreg &q) {
  return [program, obs, &q]() {
    // Observe the program
    auto programs = __internal__::observe(obs, program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }();
}

double observe(std::shared_ptr<CompositeInstruction> program, Observable &obs,
               xacc::internal_compiler::qreg &q) {
  return [program, &obs, &q]() {
    // Observe the program
    auto programs = obs.observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(&obs);
  }();
}
} // namespace __internal__


std::shared_ptr<xacc::Optimizer> createOptimizer(const std::string &type,
                                                 HeterogeneousMap &&options) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getOptimizer(type, options);
}

std::shared_ptr<xacc::Observable> createObservable(const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::quantum::getObservable("pauli", std::string(repr));
}

std::shared_ptr<Observable> createObservable(const std::string &type,
                                             const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::quantum::getObservable(type, std::string(repr));
}

std::shared_ptr<xacc::CompositeInstruction> compile(const std::string &src) {
  return xacc::getCompiler("xasm")->compile(src)->getComposites()[0];
}

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    std::function<double(const std::vector<double>,
                                         std::vector<double> &)> &&opt_function,
                    const int nParameters) {
  return std::async(std::launch::async, [=]() -> ResultsBuffer {
    qcor::OptFunction f(opt_function, nParameters);
    auto results = optimizer->optimize(f);
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &&opt_function) {
  return std::async(std::launch::async, [=, &opt_function]() -> ResultsBuffer {
    auto results = optimizer->optimize(opt_function);
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &opt_function) {
  return std::async(std::launch::async, [=, &opt_function]() -> ResultsBuffer {
    auto results = optimizer->optimize(opt_function);
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}

} // namespace qcor
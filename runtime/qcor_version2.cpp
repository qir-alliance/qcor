#include "qcor_version2.hpp"
#include "xacc.hpp"
#include "PauliOperator.hpp"
#include "Optimizer.hpp"

namespace qcor {
xacc::Optimizer *getOptimizer() {
  if (!xacc::isInitialized())
    xacc::Initialize();
  return xacc::getOptimizer("nlopt").get();
}
xacc::Observable *getObservable(const char *repr) {
  if (!xacc::isInitialized())
    xacc::Initialize();
//   auto sptr = xacc::quantum::getObservable("pauli", std::string(repr));
  auto obs = new xacc::quantum::PauliOperator(repr);
  return obs;
}

std::future<xacc::internal_compiler::qreg>
execute_algorithm(const char *objective, xacc::CompositeInstruction *program,
                  xacc::Optimizer *opt, xacc::Observable *obs, std::vector<double>& parameters) {
  return std::async(std::launch::async, [parameters, objective, opt, obs, program]() {
    auto qpu = xacc::internal_compiler::get_qpu();
    opt->appendOption("initial-parameters", parameters);
    auto algo = xacc::getAlgorithm(objective, {std::make_pair("optimizer", opt),
                                   std::make_pair("observable", obs),
                                   std::make_pair("ansatz", program),
                                   std::make_pair("accelerator", qpu)});
    auto q =
        qalloc(program->nLogicalBits());
    auto buffer = q.results();
    auto buffer_as_shared = xacc::as_shared_ptr(buffer);
    xacc::set_verbose(true);
    algo->execute(buffer_as_shared);
    delete obs;
    return q;
  });
}
template <> double extract_results<double>(xacc::internal_compiler::qreg& q, const char * key) {
    return q.results()->operator[](key).as<double>();
}
template <> double* extract_results<double*>(xacc::internal_compiler::qreg& q, const char * key) {
    // we expect that if they ask for double * it is really stored as a vector
    return q.results()->operator[](key).as<std::vector<double>>().data();
}
template <> std::vector<double> extract_results<std::vector<double>>(xacc::internal_compiler::qreg& q, const char * key) {
    // we expect that if they ask for double * it is really stored as a vector
    return q.results()->operator[](key).as<std::vector<double>>();
}
}
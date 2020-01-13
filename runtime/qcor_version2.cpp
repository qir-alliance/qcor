#include "qcor_version2.hpp"
#include "xacc.hpp"
#include "PauliOperator.hpp"
#include "Optimizer.hpp"
#include <future>

namespace qcor {
void initialize() { xacc::Initialize(); }
void finalize() {
    xacc::Finalize();
}
xacc::AcceleratorBuffer* sync(Handle& h) {return h.get();}

namespace __internal__ {

Handle execute_algorithm(const char *objective, xacc::CompositeInstruction *program,
                  xacc::Optimizer *opt, xacc::Observable *obs, double * parameters) {
  return std::async(std::launch::async, [parameters, objective, opt, obs, program]() {
    auto qpu = xacc::internal_compiler::get_qpu();
    std::vector<double> pvec(parameters, parameters + program->nVariables());
    opt->appendOption("initial-parameters", pvec);
    auto algo = xacc::getAlgorithm(objective, {std::make_pair("optimizer", opt),
                                   std::make_pair("observable", obs),
                                   std::make_pair("ansatz", program),
                                   std::make_pair("accelerator", qpu)});
    auto q = qalloc(program->nLogicalBits());
    auto buffer = q.results();
    auto buffer_as_shared = xacc::as_shared_ptr(buffer);
    xacc::set_verbose(true);
    algo->execute(buffer_as_shared);
    return buffer;
  });
  delete[] parameters;
}

}
xacc::Optimizer *getOptimizer() {
  if (!xacc::isInitialized())
    xacc::Initialize();
  return xacc::getOptimizer("nlopt").get();
}

// FIXME CLEAN UP OBS
xacc::Observable *getObservable(const char *repr) {
  if (!xacc::isInitialized())
    xacc::Initialize();
//   auto sptr = xacc::quantum::getObservable("pauli", std::string(repr));
  auto obs = new xacc::quantum::PauliOperator(repr);
  return obs;
}

template <> double extract_results<double>(xacc::AcceleratorBuffer* q, const char * key) {
    return q->operator[](key).as<double>();
}
template <> double* extract_results<double*>(xacc::AcceleratorBuffer* q, const char * key) {
    // we expect that if they ask for double * it is really stored as a vector
    return q->operator[](key).as<std::vector<double>>().data();
}
template <> std::vector<double> extract_results<std::vector<double>>(xacc::AcceleratorBuffer* q, const char * key) {
    // we expect that if they ask for double * it is really stored as a vector
    return q->operator[](key).as<std::vector<double>>();
}
}
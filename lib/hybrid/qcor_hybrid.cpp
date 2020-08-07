#include "qcor_hybrid.hpp"
#include "Compiler.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <Utils.hpp>

namespace qcor {
namespace __internal__ {

qcor::TranslationFunctor<qreg, double>
TranslationFunctorAutoGenerator::operator()(qreg &q, std::tuple<double> &&t) {
  return qcor::TranslationFunctor<qreg, double>(
      [&](const std::vector<double> x) { return std::make_tuple(q, x[0]); });
}
qcor::TranslationFunctor<qreg, std::vector<double>>
TranslationFunctorAutoGenerator::operator()(
    qreg &q, std::tuple<std::vector<double>> &&) {
  return qcor::TranslationFunctor<qreg, std::vector<double>>(
      [&](const std::vector<double> x) { return std::make_tuple(q, x); });
}
} // namespace __internal__

void QAOA::initial_compile_qaoa_code() {
  if (!xacc::isInitialized()) {
    xacc::Initialize();
  }
  auto xasm = xacc::getService<xacc::Compiler>("xasm");
  qaoa_circuit = xasm->compile(qaoa_xasm_code)->getComposites()[0];
}
void QAOA::error(const std::string &message) { xacc::error(message); }


void execute_qite(qreg q, const HeterogeneousMap &&m) {
  auto qite = xacc::getAlgorithm("qite", m);
  qite->execute(xacc::as_shared_ptr(q.results()));
}

} // namespace qcor
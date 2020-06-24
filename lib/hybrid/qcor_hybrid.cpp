#include "qcor_hybrid.hpp"

namespace qcor {
namespace __internal__ {

qcor::TranslationFunctor<qreg, double>
TranslationFunctorGenerator::operator()(qreg &q, std::tuple<double> &&t) {
  return qcor::TranslationFunctor<qreg, double>(
      [&](const std::vector<double> x) { return std::make_tuple(q, x[0]); });
}
qcor::TranslationFunctor<qreg, std::vector<double>>
TranslationFunctorGenerator::operator()(qreg &q,
                                        std::tuple<std::vector<double>> &&) {
  return qcor::TranslationFunctor<qreg, std::vector<double>>(
      [&](const std::vector<double> x) { return std::make_tuple(q, x); });
}
} // namespace __internal__

} // namespace qcor
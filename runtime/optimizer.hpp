#ifndef RUNTIME_OPTIMIZER_HPP_
#define RUNTIME_OPTIMIZER_HPP_

#include <functional>
#include <vector>

#include "InstructionParameter.hpp"
#include "Identifiable.hpp"

namespace qcor {

using OptResult = std::pair<double, std::vector<double>>;
using OptimizerOptions = std::map<std::string, xacc::InstructionParameter>;

class OptFunction {
protected:
  std::function<double(const std::vector<double> &)> _function;
  int _dim = 0;

public:
  OptFunction(std::function<double(const std::vector<double> &)> f, const int d)
      : _function(f), _dim(d) {}
  virtual const int dimensions() const { return _dim; }
  virtual double operator()(const std::vector<double> &params) {
    return _function(params);
  }
};

class Optimizer : public xacc::Identifiable {
public:
  virtual OptResult optimize(OptFunction &function,
                             OptimizerOptions options = OptimizerOptions{}) = 0;
};
} // namespace qcor
#endif

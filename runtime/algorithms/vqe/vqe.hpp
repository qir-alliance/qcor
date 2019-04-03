#ifndef RUNTIME_ALGORITHM_VQE_HPP_
#define RUNTIME_ALGORITHM_VQE_HPP_

#include "algorithm.hpp"

namespace qcor {
namespace algorithm {
class VQE : public Algorithm {
public:
  void execute(xacc::Observable &observable, Optimizer &optimizer) override;
  const std::string name() const override { return "vqe"; }
  const std::string description() const override { return ""; }
};
} // namespace algorithm
} // namespace qcor
#endif
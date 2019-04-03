#ifndef RUNTIME_ALGORITHM_HPP_
#define RUNTIME_ALGORITHM_HPP_

#include "Identifiable.hpp"
#include <memory>

namespace xacc {
class Observable;
class AcceleratorBuffer;
class Function;
class Accelerator;
} // namespace xacc

namespace qcor {

class Optimizer;
namespace algorithm {

class Algorithm : public xacc::Identifiable {

protected:
  std::shared_ptr<xacc::Function> kernel;
  std::shared_ptr<xacc::AcceleratorBuffer> buffer;
  std::shared_ptr<xacc::Accelerator> accelerator;
public:
  void initialize(std::shared_ptr<xacc::Function> k,
                   std::shared_ptr<xacc::Accelerator> acc,
                  std::shared_ptr<xacc::AcceleratorBuffer> b) {
    kernel = k;
    buffer = b;
    accelerator = acc;
  }

  virtual void execute(xacc::Observable &observable, Optimizer &optimizer) = 0;
};
} // namespace algorithm
} // namespace qcor
#endif
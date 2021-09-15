#pragma once
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "qrt.hpp"

namespace qcor {
class MirrorCircuitValidator : public BackendValidator {
public:
  virtual const std::string name() const override { return "mirror-rb"; }
  virtual const std::string description() const override { return ""; }
  virtual std::pair<bool, xacc::HeterogeneousMap>
  validate(std::shared_ptr<xacc::Accelerator> qpu,
           std::shared_ptr<qcor::CompositeInstruction> program,
           xacc::HeterogeneousMap options) override;
  // Return the 'mirrored' circuit along with the expected result.
  static std::pair<std::shared_ptr<CompositeInstruction>, std::vector<bool>>
  createMirrorCircuit(std::shared_ptr<CompositeInstruction> in_circuit);
};
} // namespace qcor

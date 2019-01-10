#ifndef IMPLS_YPULSE_HPP
#define IMPLS_YPULSE_HPP

#include "GateInstruction.hpp"

namespace xacc {
namespace quantum {
class YPulse : public GateInstruction {

public:
  YPulse();
  YPulse(std::vector<int> qbs);

  DEFINE_CLONE(YPulse)
  DEFINE_VISITABLE()

  const std::string name() const override;
  const std::string description() const override;

  const bool isAnalog() const override;
  const int nRequiredBits() const override;
  void customVisitAction(BaseInstructionVisitor &iv) override;
  virtual ~YPulse() {}
};
} // namespace quantum
} // namespace xacc

#endif

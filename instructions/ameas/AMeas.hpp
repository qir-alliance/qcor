#ifndef IMPLS_AMEAS_HPP
#define IMPLS_AMEAS_HPP

#include "GateInstruction.hpp"

namespace xacc {
namespace quantum {
class AMeas : public GateInstruction {

public:
  AMeas();
  AMeas(std::vector<int> qbs);

  DEFINE_CLONE(AMeas)
  DEFINE_VISITABLE()

  const std::string name() const override;
  const std::string description() const override;

  const bool isAnalog() const override;
  const int nRequiredBits() const override;

  void customVisitAction(BaseInstructionVisitor &iv) override;
  virtual ~AMeas() {}
};
} // namespace quantum
} // namespace xacc

#endif

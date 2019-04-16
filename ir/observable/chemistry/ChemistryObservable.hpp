#ifndef QCOR_IR_CHEMISTRY_OBSERVABLE_HPP_
#define QCOR_IR_CHEMISTRY_OBSERVABLE_HPP_

#include "Observable.hpp"

#include <thread>

using namespace xacc;

namespace xacc {
namespace quantum {
class FermionOperator;
}
} // namespace xacc
namespace qcor {

namespace observable {

class ChemistryObservable : public Observable {

protected:
  std::shared_ptr<xacc::quantum::FermionOperator> fermionOp;

public:
  std::vector<std::shared_ptr<Function>>
  observe(std::shared_ptr<Function> function) override;

  const std::string toString() override;

  void fromString(const std::string str) override;
  const int nBits() override;
  void
  fromOptions(std::map<std::string, InstructionParameter> &&options) override;
  void
  fromOptions(std::map<std::string, InstructionParameter> &options) override;

  const std::string name() const override { return "chemistry"; }
  const std::string description() const override { return ""; }
};

} // namespace observable

} // namespace qcor
#endif
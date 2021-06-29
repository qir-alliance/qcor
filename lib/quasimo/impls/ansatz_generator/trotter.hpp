#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace QuaSiMo {
// 1st-order Trotterization
class TrotterEvolution : public AnsatzGenerator {
public:
  Ansatz create_ansatz(Operator *obs,
                       const HeterogeneousMap &params) override;
  virtual const std::string name() const override { return "trotter"; }
  virtual const std::string description() const override { return ""; }
};
} // namespace QuaSiMo
} // namespace qcor
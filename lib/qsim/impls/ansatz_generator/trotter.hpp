#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace qsim {
// 1st-order Trotterization
class TrotterEvolution : public AnsatzGenerator {
public:
  Ansatz create_ansatz(Observable *obs,
                       const HeterogeneousMap &params) override;
  virtual const std::string name() const override { return "trotter"; }
  virtual const std::string description() const override { return ""; }
};
} // namespace qsim
} // namespace qcor
/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
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
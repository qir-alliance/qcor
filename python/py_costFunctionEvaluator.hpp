/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include "base/qcor_qsim.hpp"
#include <memory>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace qcor {
namespace QuaSiMo {
class PyCostFunctionEvaluator : public CostFunctionEvaluator {
  const std::string name() const override {
    PYBIND11_OVERLOAD_PURE(const std::string, CostFunctionEvaluator, name);
  }
  const std::string description() const override {
    PYBIND11_OVERLOAD_PURE(const std::string, CostFunctionEvaluator,
                           description);
  }
  double evaluate(std::shared_ptr<CompositeInstruction> state_prep) override {
    PYBIND11_OVERLOAD_PURE(double, CostFunctionEvaluator, evaluate);
  }
  bool initialize(Operator *observable,
                  const HeterogeneousMap &params) override {
    PYBIND11_OVERLOAD_PURE(bool, CostFunctionEvaluator, initialize);
  }
};
} // namespace QuaSiMo
} // namespace qcor

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
namespace qsim {
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
  bool initialize(Observable *observable,
                  const HeterogeneousMap &params) override {
    PYBIND11_OVERLOAD_PURE(bool, CostFunctionEvaluator, initialize);
  }
};
} // namespace qsim
} // namespace qcor

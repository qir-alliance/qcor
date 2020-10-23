#include "base/qcor_qsim.hpp"
#include "py_costFunctionEvaluator.hpp"
#include "py_qsimWorkflow.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace pybind11 {
namespace detail {
template <typename... Ts>
struct type_caster<xacc::Variant<Ts...>>
    : variant_caster<xacc::Variant<Ts...>> {};

template <> struct visit_helper<xacc::Variant> {
  template <typename... Args>
  static auto call(Args &&... args) -> decltype(mpark::visit(args...)) {
    return mpark::visit(args...);
  }
};

template <typename... Ts>
struct type_caster<mpark::variant<Ts...>>
    : variant_caster<mpark::variant<Ts...>> {};

template <> struct visit_helper<mpark::variant> {
  template <typename... Args>
  static auto call(Args &&... args) -> decltype(mpark::visit(args...)) {
    return mpark::visit(args...);
  }
};
} // namespace detail
} // namespace pybind11

namespace {
// Add type name to this list to support receiving from Python.
using PyHeterogeneousMapTypes = xacc::Variant<bool, int, double, std::string>;

using PyHeterogeneousMap = std::map<std::string, PyHeterogeneousMapTypes>;

// Helper to convert a Python *dict* (as a map of variants) into a native
// HetMap.
xacc::HeterogeneousMap
heterogeneousMapConvert(const PyHeterogeneousMap &in_pyMap) {
  xacc::HeterogeneousMap result;
  for (auto &item : in_pyMap) {
    auto visitor = [&](const auto &value) { result.insert(item.first, value); };
    mpark::visit(visitor, item.second);
  }

  return result;
}
} // namespace

PYBIND11_MODULE(_pyqcor, m) {
  m.doc() = "Python bindings for QCOR.";
  // Handle QCOR CLI arguments:
  // when using via Python, we use this to set those runtime parameters.
  m.def(
      "Initialize",
      [](py::kwargs kwargs) {
        if (kwargs) {
          for (auto arg : kwargs) {
            const auto key = std::string(py::str(arg.first));
            const auto value = std::string(py::str(arg.second));
            // Handle "qpu" key
            if (key == "qpu") {
              quantum::initialize(value, "empty");
            }
            /// TODO: handle other CLI parameters.
          }
        }
      },
      "Initialize QCOR runtime environment.");

  // Expose QCOR API functions
  m.def(
      "createOptimizer",
      [](const std::string &name, PyHeterogeneousMap p = {}) {
        return qcor::createOptimizer(name, heterogeneousMapConvert(p));
      },
      py::arg("name"), py::arg("p") = PyHeterogeneousMap(),
      py::return_value_policy::reference,
      "Return the Optimizer with given name.");

  // qsim sub-module bindings:
  {
    py::module qsim = m.def_submodule("qsim", "QCOR's python qsim submodule");

    // QuantumSimulationModel bindings:
    py::class_<qcor::qsim::QuantumSimulationModel>(
        qsim, "QuantumSimulationModel",
        "The QuantumSimulationModel captures the quantum simulation problem "
        "description.")
        .def(py::init<>());

    // ModelBuilder bindings:
    py::class_<qcor::qsim::ModelBuilder>(
        qsim, "ModelBuilder",
        "The ModelBuilder interface provides methods to "
        "construct qsim problem models.")
        .def(py::init<>())
        .def(
            "createModel",
            [](qcor::PauliOperator &obs, qcor::qsim::TdObservable ham_func) {
              return qcor::qsim::ModelBuilder::createModel(obs, ham_func);
            },
            "Return the Model for a time-dependent problem.");

    // CostFunctionEvaluator bindings
    py::class_<qcor::qsim::CostFunctionEvaluator,
               std::shared_ptr<qcor::qsim::CostFunctionEvaluator>,
               qcor::qsim::PyCostFunctionEvaluator>(
        qsim, "CostFunctionEvaluator",
        "The CostFunctionEvaluator interface provides methods to "
        "evaluate the observable operator expectation value on quantum "
        "backends.")
        .def(py::init<>())
        .def(
            "initialize",
            [](qcor::qsim::CostFunctionEvaluator &self,
               qcor::PauliOperator &obs) { return self.initialize(&obs); },
            "Initialize the evaluator")
        .def(
            "evaluate",
            [](qcor::qsim::CostFunctionEvaluator &self,
               std::shared_ptr<CompositeInstruction> state_prep) -> double {
              return self.evaluate(state_prep);
            },
            "Initialize the evaluator");
    qsim.def(
        "getObjEvaluator",
        [](qcor::PauliOperator &obs, const std::string &name = "default",
           py::dict p = {}) { return qcor::qsim::getObjEvaluator(obs, name); },
        py::arg("obs"), py::arg("name") = "default", py::arg("p") = py::dict(),
        py::return_value_policy::reference,
        "Return the CostFunctionEvaluator.");

    // QuantumSimulationWorkflow bindings
    py::class_<qcor::qsim::QuantumSimulationWorkflow,
               std::shared_ptr<qcor::qsim::QuantumSimulationWorkflow>,
               qcor::qsim::PyQuantumSimulationWorkflow>(
        qsim, "QuantumSimulationWorkflow",
        "The QuantumSimulationWorkflow interface provides methods to "
        "execute a quantum simulation workflow.")
        .def(py::init<>())
        .def(
            "execute",
            [](qcor::qsim::QuantumSimulationWorkflow &self,
               const qcor::qsim::QuantumSimulationModel &model)
                -> qcor::qsim::QuantumSimulationResult {
              return self.execute(model);
            },
            "Execute the workflow for the input problem model.");
    qsim.def(
        "getWorkflow",
        [](const std::string &name, PyHeterogeneousMap p = {}) {
          auto nativeHetMap = heterogeneousMapConvert(p);
          return qcor::qsim::getWorkflow(name, nativeHetMap);
        },
        py::arg("name"), py::arg("p") = PyHeterogeneousMap(),
        py::return_value_policy::reference,
        "Return the quantum simulation workflow.");
  }
}

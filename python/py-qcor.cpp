#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "base/qcor_qsim.hpp"
#include "py_costFunctionEvaluator.hpp"
#include "py_qsimWorkflow.hpp"
#include "qcor_jit.hpp"
#include "qcor_observable.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

namespace py = pybind11;
using namespace xacc;

namespace pybind11 {
namespace detail {
template <typename... Ts>
struct type_caster<Variant<Ts...>> : variant_caster<Variant<Ts...>> {};

template <>
struct visit_helper<Variant> {
  template <typename... Args>
  static auto call(Args &&... args) -> decltype(mpark::visit(args...)) {
    return mpark::visit(args...);
  }
};

template <typename... Ts>
struct type_caster<mpark::variant<Ts...>>
    : variant_caster<mpark::variant<Ts...>> {};

template <>
struct visit_helper<mpark::variant> {
  template <typename... Args>
  static auto call(Args &&... args) -> decltype(mpark::visit(args...)) {
    return mpark::visit(args...);
  }
};
}  // namespace detail
}  // namespace pybind11

namespace {

// We only allow certain argument types for quantum kernel functors in python
// Here we enumerate them as a Variant
using AllowedKernelArgTypes =
    xacc::Variant<bool, int, double, std::string, xacc::internal_compiler::qreg,
                  std::vector<double>>;

// We will take as input a mapping of arg variable names to the argument itself.
using KernelArgDict = std::map<std::string, AllowedKernelArgTypes>;

// Utility for mapping KernelArgDict to a HeterogeneousMap
class KernelArgDictToHeterogeneousMap {
 protected:
  xacc::HeterogeneousMap &m;
  const std::string &key;

 public:
  KernelArgDictToHeterogeneousMap(xacc::HeterogeneousMap &map,
                                  const std::string &k)
      : m(map), key(k) {}
  template <typename T>
  void operator()(const T &t) {
    m.insert(key, t);
  }
};

// Add type name to this list to support receiving from Python.
using PyHeterogeneousMapTypes = xacc::Variant<bool, int, double, std::string,
                                              std::shared_ptr<qcor::Optimizer>>;
using PyHeterogeneousMap = std::map<std::string, PyHeterogeneousMapTypes>;

// Helper to convert a Python *dict* (as a map of variants) into a native
// HetMap.
xacc::HeterogeneousMap heterogeneousMapConvert(
    const PyHeterogeneousMap &in_pyMap) {
  xacc::HeterogeneousMap result;
  for (auto &item : in_pyMap) {
    auto visitor = [&](const auto &value) { result.insert(item.first, value); };
    mpark::visit(visitor, item.second);
  }

  return result;
}
}  // namespace

namespace qcor {

// PyObjectiveFunction implements ObjectiveFunction to 
// enable the utility of pythonic quantum kernels with the 
// existing qcor ObjectiveFunction infrastructure. This class 
// keeps track of the quantum kernel as a py::object, which it uses 
// in tandem with the QCOR QJIT engine to create an executable 
// functor representation of the quantum code at runtime. It exposes 
// the ObjectiveFunction operator()() overloads to map vector<double> 
// x to the correct pythonic argument structure. It delegates to the 
// usual helper ObjectiveFunction (like vqe) for execution of the 
// actual pre-, execution, and post-processing. 
class PyObjectiveFunction : public qcor::ObjectiveFunction {
 protected:
  py::object py_kernel;
  std::shared_ptr<ObjectiveFunction> helper;
  xacc::internal_compiler::qreg qreg;
  QJIT qjit;

 public:
  const std::string name() const override { return "py-objective-impl"; }
  const std::string description() const override { return ""; }
  PyObjectiveFunction(py::object q, qcor::PauliOperator &qq, const int n_dim,
                      const std::string &helper_name)
      : py_kernel(q) {
    
    // Set the OptFunction dimensions
    _dim = n_dim;

    // Set the helper objective
    helper = xacc::getService<qcor::ObjectiveFunction>(helper_name);

    // Store the observable pointer and give it to the helper
    observable = xacc::as_shared_ptr(&qq);
    helper->update_observable(observable);

    // Extract the QJIT source code
    auto src = py_kernel.attr("get_internal_src")().cast<std::string>();

    // QJIT compile
    // this will be fast if already done, and we just do it once
    qjit.jit_compile(src, true);
  }

  // Evaluate this ObjectiveFunction at the dictionary of kernel args, 
  // return the scalar value 
  double operator()(const KernelArgDict args) {
    // Map the kernel args to a hetmap
    xacc::HeterogeneousMap m;
    for (auto &item : args) {
      KernelArgDictToHeterogeneousMap vis(m, item.first);
      mpark::visit(vis, item.second);
    }

    // Get the kernel as a CompositeInstruction
    auto kernel_name = py_kernel.attr("kernel_name")().cast<std::string>();
    kernel = qjit.extract_composite_with_hetmap(kernel_name, m);
    helper->update_kernel(kernel);

    // FIXME, handle gradients
    std::vector<double> dx;
    return (*helper)(qreg, dx);
  }

  // Evaluate this ObjectiveFunction at the parameters x
  double operator()(const std::vector<double> &x,
                    std::vector<double> &dx) override {
    current_iterate_parameters = x;
    helper->update_current_iterate_parameters(x);

    // Translate x into kernel args
    qreg = ::qalloc(observable->nBits());
    auto args = py_kernel.attr("translate")(qreg, x).cast<KernelArgDict>();
    // args will be a dictionary, arg_name to arg
    return operator()(args);
  }

  virtual double operator()(xacc::internal_compiler::qreg &qreg,
                            std::vector<double> &dx) {
    throw std::bad_function_call();
    return 0.0;
  }
};

// PyKernelFunctor is a subtype of KernelFunctor from the qsim library 
// that returns a CompositeInstruction representation of a pythonic 
// quantum kernel given a vector of parameters x. This will 
// leverage the QJIT infrastructure to create executable functor 
// representation of the python kernel. 
class PyKernelFunctor : public qcor::KernelFunctor {
 protected:
  py::object py_kernel;
  QJIT qjit;
  std::size_t n_qubits;

 public:
  PyKernelFunctor(py::object q, const std::size_t nq, const std::size_t np)
      : py_kernel(q), n_qubits(nq) {
    nbParams = np;
    auto src = py_kernel.attr("get_internal_src")().cast<std::string>();
    // this will be fast if already done, and we just do it once
    qjit.jit_compile(src, true);
  }

  // Delegate to QJIT to create a CompositeInstruction representation 
  // of the pythonic quantum kernel. 
  std::shared_ptr<xacc::CompositeInstruction> evaluate_kernel(
      const std::vector<double> &x) override {
    // Translate x into kernel args
    auto qreg = ::qalloc(n_qubits);
    auto args = py_kernel.attr("translate")(qreg, x).cast<KernelArgDict>();
    xacc::HeterogeneousMap m;
    for (auto &item : args) {
      KernelArgDictToHeterogeneousMap vis(m, item.first);
      mpark::visit(vis, item.second);
    }
    auto kernel_name = py_kernel.attr("kernel_name")().cast<std::string>();
    return qjit.extract_composite_with_hetmap(kernel_name, m);
  }
};
}  // namespace qcor

PYBIND11_MODULE(_pyqcor, m) {
  m.doc() = "Python bindings for QCOR.";

  py::class_<AllowedKernelArgTypes>(
      m, "AllowedKernelArgTypes",
      "The AllowedKernelArgTypes provides a variant structure "
      "to provide parameters to qcor quantum kernels HeterogeneousMaps.")
      .def(py::init<int>(), "Construct as an int.")
      .def(py::init<bool>(), "Construct as a bool")
      .def(py::init<double>(), "Construct as a double.")
      .def(py::init<std::string>(), "Construct as a string.")
      .def(py::init<xacc::internal_compiler::qreg>(), "Construct as qreg")
      .def(py::init<std::vector<double>>(), "Construct as a List[double].");

  // Expose QCOR API functions
  // Handle QCOR CLI arguments:
  // when using via Python, we use this to set those runtime parameters.
  m.def(
      "Initialize",
      [](py::kwargs kwargs) {
        if (kwargs) {
          for (auto arg : kwargs) {
            const auto key = std::string(py::str(arg.first));
            // Handle "qpu" key
            if (key == "qpu") {
              const auto value = std::string(py::str(arg.second));
              ::quantum::initialize(value, "empty");
            } else if (key == "shots") {
              const auto value = arg.second.cast<int>();
              ::quantum::set_shots(value);
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

  m.def(
      "set_qpu",
      [](const std::string &name, PyHeterogeneousMap p = {}) {
        xacc::internal_compiler::qpu =
            xacc::getAccelerator(name, heterogeneousMapConvert(p));
      },
      py::arg("name"), py::arg("p") = PyHeterogeneousMap(),
      "Set the QPU backend.");

  m.def("qalloc", &::qalloc, py::return_value_policy::reference, "");
  py::class_<xacc::internal_compiler::qreg>(m, "qreg", "")
      .def("size", &xacc::internal_compiler::qreg::size, "")
      .def("print", &xacc::internal_compiler::qreg::print, "")
      .def("counts", &xacc::internal_compiler::qreg::counts, "")
      .def("exp_val_z", &xacc::internal_compiler::qreg::exp_val_z, "");

  // m.def("createObjectiveFunction", [](const std::string name, ))
  py::class_<qcor::QJIT, std::shared_ptr<qcor::QJIT>>(m, "QJIT", "")
      .def(py::init<>(), "")
      .def("jit_compile", &qcor::QJIT::jit_compile, "")
      .def(
          "internal_python_jit_compile",
          [](qcor::QJIT &qjit, const std::string src) {
            bool turn_on_hetmap_kernel_ctor = true;
            qjit.jit_compile(src, turn_on_hetmap_kernel_ctor);
          },
          "")
      .def("run_syntax_handler", &qcor::QJIT::run_syntax_handler, "")
      .def(
          "invoke",
          [](qcor::QJIT &qjit, const std::string name, KernelArgDict args) {
            xacc::HeterogeneousMap m;
            for (auto &item : args) {
              KernelArgDictToHeterogeneousMap vis(m, item.first);
              mpark::visit(vis, item.second);
            }
            qjit.invoke_with_hetmap(name, m);
          },
          "")
      .def("extract_composite",
           [](qcor::QJIT &qjit, const std::string name, KernelArgDict args) {
             xacc::HeterogeneousMap m;
             for (auto &item : args) {
               KernelArgDictToHeterogeneousMap vis(m, item.first);
               mpark::visit(vis, item.second);
             }
             return qjit.extract_composite_with_hetmap(name, m);
           });

  py::class_<qcor::ObjectiveFunction, std::shared_ptr<qcor::ObjectiveFunction>>(
      m, "ObjectiveFunction", "")
      .def("dimensions", &qcor::ObjectiveFunction::dimensions, "")
      .def(
          "__call__",
          [](qcor::ObjectiveFunction &obj, std::vector<double> x) {
            return obj(x);
          },
          "");

  m.def(
      "createObjectiveFunction",
      [](py::object kernel, qcor::PauliOperator &obs, const int n_params) {
        auto q = ::qalloc(obs.nBits());
        std::shared_ptr<qcor::ObjectiveFunction> obj =
            std::make_shared<qcor::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe");
        return obj;
      },
      "");

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
            "Return the Model for a time-dependent problem.")
        .def(
            "createModel",
            [](py::object py_kernel, qcor::PauliOperator &obs,
               const int n_qubits, const int n_params) {
              qcor::qsim::QuantumSimulationModel model;
              auto kernel_functor = std::make_shared<qcor::PyKernelFunctor>(
                  py_kernel, n_qubits, n_params);
              model.observable = &obs;
              model.user_defined_ansatz = kernel_functor;
              return std::move(model);
            },
            "");

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

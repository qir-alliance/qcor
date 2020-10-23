#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "base/qcor_qsim.hpp"
#include "qcor_jit.hpp"
#include "qrt.hpp"

#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"

namespace py = pybind11;
using namespace xacc;

namespace pybind11 {
namespace detail {
template <typename... Ts>
struct type_caster<Variant<Ts...>> : variant_caster<Variant<Ts...>> {};

template <> struct visit_helper<Variant> {
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


// We can't pass HeterogeneousMap back and forth effectively,
// so here we define a variant of common input types and an
// associated map to fake like it is a HeterogeneousMap
using AllowedKernelArgTypes =
    xacc::Variant<bool, int, double, std::string, std::vector<std::string>,
                  xacc::internal_compiler::qreg, std::vector<double>,
                  std::vector<int>, std::complex<double>,
                  std::vector<std::complex<double>>, Eigen::MatrixXcd>;
using KernelArgDict = std::map<std::string, AllowedKernelArgTypes>;

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
      .def(py::init<std::vector<std::string>>(), "Construct as a List[string].")
      .def(py::init<std::vector<int>>(), "Construct as a List[int].")
      .def(py::init<std::vector<double>>(), "Construct as a List[double].")
      .def(py::init<std::vector<std::complex<double>>>(),
           "Construct as a List[complex].")
      .def(py::init<Eigen::MatrixXcd>(), "Construct as an Eigen matrix.");

  // Expose QCOR API functions
  m.def(
      "createOptimizer",
      [](const std::string &name, py::dict p = {}) {
        xacc::HeterogeneousMap params;
        /// TODO: reuse XACC PyHetMap here
        for (auto item : p) {
          std::cout << "key=" << std::string(py::str(item.first)) << ", "
                    << "value=" << std::string(py::str(item.second)) << "\n";
        }

        return qcor::createOptimizer(name, std::move(params));
      },
      py::arg("name"), py::arg("p") = py::dict(),
      py::return_value_policy::reference,
      "Return the Optimizer with given name.");

  m.def("qrt_initialize", &::quantum::initialize, "");
  m.def(
      "set_qpu",
      [](const std::string &name) {
        xacc::internal_compiler::qpu = xacc::getAccelerator(name);
      },
      "Set the QPU backend.");

  m.def("qalloc", &::qalloc, py::return_value_policy::reference, "");
  py::class_<xacc::internal_compiler::qreg>(m, "qreg", "")
      .def("size", &xacc::internal_compiler::qreg::size, "")
      .def("print", &xacc::internal_compiler::qreg::print, "");

  py::class_<qcor::QJIT, std::shared_ptr<qcor::QJIT>>(m, "QJIT", "")
      .def(py::init<>(), "")
      .def("jit_compile", &qcor::QJIT::jit_compile, "")
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
            "Return the Model for a time-dependent problem.");
  }
}

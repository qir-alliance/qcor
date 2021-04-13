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
#ifdef QCOR_BUILD_MLIR_PYTHON_API
#include "qcor_mlir_api.hpp"
#endif 

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
  static auto call(Args &&...args) -> decltype(mpark::visit(args...)) {
    return mpark::visit(args...);
  }
};

template <typename... Ts>
struct type_caster<mpark::variant<Ts...>>
    : variant_caster<mpark::variant<Ts...>> {};

template <>
struct visit_helper<mpark::variant> {
  template <typename... Args>
  static auto call(Args &&...args) -> decltype(mpark::visit(args...)) {
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
                  std::vector<double>, std::vector<int>, qcor::PauliOperator,
                  qcor::FermionOperator, qcor::PairList<int>,
                  std::vector<qcor::PauliOperator>,
                  std::vector<qcor::FermionOperator>>;

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
using PyHeterogeneousMapTypes =
    xacc::Variant<bool, int, double, std::string, std::vector<int>,
                  std::vector<std::pair<int, int>>,
                  std::shared_ptr<qcor::Optimizer>, std::vector<double>,
                  std::vector<std::vector<double>>>;
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

std::shared_ptr<qcor::Observable> convertToQCOROperator(
    py::object op, bool keep_fermion = false) {
  if (py::hasattr(op, "terms")) {
    // this is from openfermion
    if (py::hasattr(op, "is_two_body_number_conserving")) {
      // This is a fermion Operator
      auto terms = op.attr("terms");
      // terms is a list of tuples
      std::stringstream ss;
      int i = 0;
      for (auto term : terms) {
        auto term_tuple = term.cast<py::tuple>();
        if (!term_tuple.empty()) {
          ss << terms[term].cast<std::complex<double>>() << " ";
          for (auto element : term_tuple) {
            auto element_pair = element.cast<std::pair<int, int>>();
            ss << element_pair.first << (element_pair.second ? "^" : "") << " ";
          }
        } else {
          // this was identity
          try {
            auto coeff = terms[term].cast<double>();
            ss << coeff;
          } catch (std::exception &e) {
            try {
              auto coeff = terms[term].cast<std::complex<double>>();
              ss << coeff;
            } catch (std::exception &e) {
              qcor::error(
                  "Could not cast identity coefficient to double or complex.");
            }
          }
        }
        i++;
        if (i != py::len(terms)) {
          ss << " + ";
        }
      }
      auto obs_tmp = qcor::createOperator("fermion", ss.str());
      if (keep_fermion) {
        return obs_tmp;
      } else {
        return qcor::operatorTransform("jw", obs_tmp);
      }

    } else {
      if (keep_fermion) {
        xacc::error(
            "Error - you asked for a qcor::FermionOperator, but this is an "
            "OpenFermion QubitOperator.");
      }
      // this is a qubit  operator
      auto terms = op.attr("terms");
      // terms is a list of tuples
      std::stringstream ss;
      int i = 0;
      for (auto term : terms) {
        auto term_tuple = term.cast<py::tuple>();
        if (!term_tuple.empty()) {
          ss << terms[term].cast<std::complex<double>>() << " ";
          for (auto element : term_tuple) {
            auto element_pair = element.cast<std::pair<int, std::string>>();
            ss << element_pair.second << element_pair.first << " ";
          }
        } else {
          // this was identity

          try {
            auto coeff = terms[term].cast<double>();
            ss << coeff;
          } catch (std::exception &e) {
            try {
              auto coeff = terms[term].cast<std::complex<double>>();
              ss << coeff;
            } catch (std::exception &e) {
              qcor::error(
                  "Could not cast identity coefficient to double or complex.");
            }
          }
        }
        i++;
        if (i != py::len(terms)) {
          ss << " + ";
        }
      }
      return qcor::createOperator(ss.str());
    }
  } else if (py::hasattr(op, "toString") && py::hasattr(op, "observe")) {
    auto string_rep = op.attr("toString");
    auto op_str = string_rep().cast<std::string>();
    if (op_str.find("^") != std::string::npos) {
      return qcor::createOperator("fermion", op_str);
    } else {
      return qcor::createOperator(op_str);
    }
  } else {
    // throw an error
    qcor::error(
        "Invalid python object passed as a QCOR Operator/Observable. "
        "Currently, we only accept OpenFermion datastructures.");
    return nullptr;
  }
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
                      const std::string &helper_name,
                      xacc::HeterogeneousMap opts = {})
      : py_kernel(q) {
    // Set the OptFunction dimensions
    _dim = n_dim;

    qreg = ::qalloc(qq.nBits());

    // Set the helper objective
    helper = xacc::getService<qcor::ObjectiveFunction>(helper_name);

    // Store the observable pointer and give it to the helper
    observable = xacc::as_shared_ptr(&qq);
    options = opts;
    options.insert("observable", observable);
    helper->set_options(options);
    helper->update_observable(observable);

    // Extract the QJIT source code
    auto src = py_kernel.attr("get_internal_src")().cast<std::string>();
    auto extra_cpp_src =
        py_kernel.attr("get_extra_cpp_code")().cast<std::string>();
    auto sorted_kernel_deps = py_kernel.attr("get_sorted_kernels_deps")()
                                  .cast<std::vector<std::string>>();

    // QJIT compile
    // this will be fast if already done, and we just do it once
    qjit.jit_compile(src, true, sorted_kernel_deps, extra_cpp_src);
    qjit.write_cache();
  }

  PyObjectiveFunction(py::object q, std::shared_ptr<qcor::Observable> &qq,
                      const int n_dim, const std::string &helper_name,
                      xacc::HeterogeneousMap opts = {})
      : py_kernel(q) {
    // Set the OptFunction dimensions
    _dim = n_dim;
    qreg = ::qalloc(qq->nBits());

    // Set the helper objective
    helper = xacc::getService<qcor::ObjectiveFunction>(helper_name);

    // Store the observable pointer and give it to the helper
    observable = qq;
    options = opts;
    options.insert("observable", observable);
    helper->set_options(options);
    helper->update_observable(observable);

    // Extract the QJIT source code
    auto src = py_kernel.attr("get_internal_src")().cast<std::string>();
    auto extra_cpp_src =
        py_kernel.attr("get_extra_cpp_code")().cast<std::string>();
    auto sorted_kernel_deps = py_kernel.attr("get_sorted_kernels_deps")()
                                  .cast<std::vector<std::string>>();

    // QJIT compile
    // this will be fast if already done, and we just do it once
    qjit.jit_compile(src, true, sorted_kernel_deps, extra_cpp_src);
    qjit.write_cache();
  }
  // Evaluate this ObjectiveFunction at the dictionary of kernel args,
  // return the scalar value
  double operator()(const KernelArgDict args, std::vector<double> &dx) {
    std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>
        kernel_evaluator = [&](std::vector<double> x) {
          // qreg = ::qalloc(observable->nBits());
          // std::cout << "Allocating " << qreg.name() << "\n";
          auto _args =
              py_kernel.attr("translate")(qreg, x).cast<KernelArgDict>();
          // Map the kernel args to a hetmap
          xacc::HeterogeneousMap m;
          for (auto &item : _args) {
            KernelArgDictToHeterogeneousMap vis(m, item.first);
            mpark::visit(vis, item.second);
          }

          // Get the kernel as a CompositeInstruction
          auto kernel_name =
              py_kernel.attr("kernel_name")().cast<std::string>();
          return qjit.extract_composite_with_hetmap(kernel_name, m);
        };

    kernel = kernel_evaluator(current_iterate_parameters);
    helper->update_kernel(kernel);
    helper->update_options("kernel-evaluator", kernel_evaluator);

    return (*helper)(qreg, dx);
  }

  // Evaluate this ObjectiveFunction at the parameters x
  double operator()(const std::vector<double> &x,
                    std::vector<double> &dx) override {
    current_iterate_parameters = x;
    helper->update_current_iterate_parameters(x);

    // Translate x into kernel args
    // qreg = ::qalloc(observable->nBits());
    auto args = py_kernel.attr("translate")(qreg, x).cast<KernelArgDict>();
    // args will be a dictionary, arg_name to arg
    return operator()(args, dx);
  }

  virtual double operator()(xacc::internal_compiler::qreg &qreg,
                            std::vector<double> &dx) {
    throw std::bad_function_call();
    return 0.0;
  }

  xacc::internal_compiler::qreg get_qreg() override { return qreg; }
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
    auto extra_cpp_src =
        py_kernel.attr("get_extra_cpp_code")().cast<std::string>();
    auto sorted_kernel_deps = py_kernel.attr("get_sorted_kernels_deps")()
                                  .cast<std::vector<std::string>>();

    // this will be fast if already done, and we just do it once
    qjit.jit_compile(src, true, sorted_kernel_deps, extra_cpp_src);
    qjit.write_cache();
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
          // QRT (if provided) must be set before quantum::initialize
          if (kwargs.contains("qrt")) {
            const auto value = std::string(py::str(kwargs["qrt"]));
            // QRT (if provided) should be set before quantum::initialize
            ::quantum::set_qrt(value);
          }

          for (auto arg : kwargs) {
            const auto key = std::string(py::str(arg.first));
            // Handle "qpu" key
            if (key == "qpu") {
              const auto value = std::string(py::str(arg.second));
              ::quantum::initialize(value, "empty");
            } else if (key == "shots") {
              const auto value = arg.second.cast<int>();
              ::quantum::set_shots(value);
            } else if (key == "opt") {
              const auto value = arg.second.cast<int>();
              xacc::internal_compiler::__opt_level = value;
            } else if (key == "print-opt-stats") {
              const auto value = arg.second.cast<bool>();
              xacc::internal_compiler::__print_opt_stats = value;
            } else if (key == "placement") {
              const auto value = std::string(py::str(arg.second));
              xacc::internal_compiler::__placement_name = value;
            } else if (key == "opt-pass") {
              const auto value = std::string(py::str(arg.second));
              xacc::internal_compiler::__user_opt_passes = value;
            } else if (key == "qubit-map") {
              const auto value = std::string(py::str(arg.second));
              xacc::internal_compiler::__qubit_map =
                  xacc::internal_compiler::parse_qubit_map(value.c_str());
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

  m.def(
      "set_opt_level",
      [](int level) { xacc::internal_compiler::__opt_level = level; },
      py::arg("level"), "Set QCOR runtime optimization level.");

  m.def(
      "add_pass",
      [](const std::string &pass_name) {
        // Note: we expect __user_opt_passes to be a comma-separated list of
        // pass names.
        if (xacc::internal_compiler::__user_opt_passes.empty()) {
          xacc::internal_compiler::__user_opt_passes = pass_name;
        } else {
          xacc::internal_compiler::__user_opt_passes += ("," + pass_name);
        }
      },
      py::arg("pass_name"),
      "Add an optimization pass to be run by the PassManager.");

  m.def(
      "get_placement_names",
      []() {
        std::vector<std::string> result;
        auto ir_transforms = xacc::getServices<xacc::IRTransformation>();
        for (const auto &plugin : ir_transforms) {
          if (plugin->type() == xacc::IRTransformationType::Placement) {
            result.emplace_back(plugin->name());
          }
        }
        return result;
      },
      "Get names of all available placement plugins.");

  m.def(
      "set_placement",
      [](const std::string &placement_name) {
        xacc::internal_compiler::__placement_name = placement_name;
      },
      py::arg("placement_name"), "Set the placement strategy.");

  m.def("qalloc", &::qalloc, py::return_value_policy::reference, "");
  py::class_<xacc::internal_compiler::qubit>(m, "qubit", "");
  py::class_<xacc::internal_compiler::qreg>(m, "qreg", "")
      .def("size", &xacc::internal_compiler::qreg::size, "")
      .def("print", &xacc::internal_compiler::qreg::print, "")
      .def("counts", &xacc::internal_compiler::qreg::counts, "")
      .def("extract_range", [](xacc::internal_compiler::qreg& q, std::size_t start, std::size_t end){
        std::vector<std::size_t> r{start, end};
        return q.extract_range(r);
      }, "")
      // .def("extract_qubits", &xacc::internal_compiler::qreg::extract_qubits, "")
      .def("exp_val_z", &xacc::internal_compiler::qreg::exp_val_z, "")
      .def("results", [](xacc::internal_compiler::qreg& q){
        auto buffer = q.results_shared();
        return buffer;
      }, "")
      .def(
          "getInformation",
          [](xacc::internal_compiler::qreg &q, const std::string &key) {
            return q.results()->getInformation(key);
          },
          "");
  // m.def("createObjectiveFunction", [](const std::string name, ))
  py::class_<qcor::QJIT, std::shared_ptr<qcor::QJIT>>(m, "QJIT", "")
      .def(py::init<>(), "")
      .def("write_cache", &qcor::QJIT::write_cache, "")
      .def(
          "jit_compile",
          [](qcor::QJIT &qjit, const std::string src) {
            bool turn_on_hetmap_kernel_ctor = true;
            qjit.jit_compile(src, turn_on_hetmap_kernel_ctor, {});
          },
          "")
      .def(
          "internal_python_jit_compile",
          [](qcor::QJIT &qjit, const std::string src,
             const std::vector<std::string> &dependency = {},
             const std::string &extra_cpp_code = "",
             std::vector<std::string> extra_headers = {}) {
            bool turn_on_hetmap_kernel_ctor = true;
            qjit.jit_compile(src, turn_on_hetmap_kernel_ctor, dependency,
                             extra_cpp_code, extra_headers);
          },
          "")
      .def(
          "run_syntax_handler",
          [](qcor::QJIT &qjit, const std::string src) {
            return qjit.run_syntax_handler(src, true);
          },
          "")
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
           })
      .def("internal_as_unitary",
           [](qcor::QJIT &qjit, const std::string name, KernelArgDict args) {
             xacc::HeterogeneousMap m;
             for (auto &item : args) {
               KernelArgDictToHeterogeneousMap vis(m, item.first);
               mpark::visit(vis, item.second);
             }
             auto composite = qjit.extract_composite_with_hetmap(name, m);
             auto n_qubits = composite->nLogicalBits();
             qcor::KernelToUnitaryVisitor visitor(n_qubits);
             InstructionIterator iter(composite);
             while (iter.hasNext()) {
               auto inst = iter.next();
               if (!inst->isComposite() && inst->isEnabled()) {
                 inst->accept(&visitor);
               }
             }
             return visitor.getMat();
           })
      .def(
          "get_kernel_function_ptr",
          [](qcor::QJIT &qjit, const std::string &kernel_name) {
            return qjit.get_kernel_function_ptr(kernel_name);
          },
          "");

  py::class_<qcor::ObjectiveFunction, std::shared_ptr<qcor::ObjectiveFunction>>(
      m, "ObjectiveFunction", "")
      .def("dimensions", &qcor::ObjectiveFunction::dimensions, "")
      .def(
          "__call__",
          [](qcor::ObjectiveFunction &obj, std::vector<double> x) {
            return obj(x);
          },
          "")
      .def(
          "__call__",
          [](qcor::ObjectiveFunction &obj, std::vector<double> x,
             std::vector<double> &dx) {
            auto val = obj(x, dx);
            return std::make_pair(val, dx);
          },
          "")
      .def("get_qreg", &qcor::ObjectiveFunction::get_qreg, "");

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
  m.def(
      "createObjectiveFunction",
      [](py::object kernel, py::object &py_obs, const int n_params) {
        auto obs = convertToQCOROperator(py_obs);
        auto q = ::qalloc(obs->nBits());
        std::shared_ptr<qcor::ObjectiveFunction> obj =
            std::make_shared<qcor::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe");
        return obj;
      },
      "");
  m.def(
      "createObjectiveFunction",
      [](py::object kernel, qcor::PauliOperator &obs, const int n_params,
         PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        auto q = ::qalloc(obs.nBits());
        std::shared_ptr<qcor::ObjectiveFunction> obj =
            std::make_shared<qcor::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe", nativeHetMap);
        return obj;
      },
      "");
  m.def(
      "createObjectiveFunction",
      [](py::object kernel, py::object &py_obs, const int n_params,
         PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        auto obs = convertToQCOROperator(py_obs);
        auto q = ::qalloc(obs->nBits());
        std::shared_ptr<qcor::ObjectiveFunction> obj =
            std::make_shared<qcor::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe", nativeHetMap);
        return obj;
      },
      "");

  m.def(
      "createOperator",
      [](const std::string &repr) { return qcor::createOperator(repr); }, "");
  m.def(
      "createOperator",
      [](const std::string &type, const std::string &repr) {
        auto op = qcor::createOperator(type, repr);
      },
      "");
  m.def(
      "createOperator",
      [](const std::string &type, PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        return qcor::createOperator(type, nativeHetMap);
      },
      "");
  m.def("createOperator", [](const std::string &type, py::object pyobject) {
    return convertToQCOROperator(pyobject, type == "fermion");
  });
  m.def(
      "createObservable",
      [](const std::string &repr) { return qcor::createOperator(repr); }, "");
  m.def(
      "createObservable",
      [](const std::string &type, const std::string &repr) {
        return qcor::createOperator(type, repr);
      },
      "");
  m.def(
      "createObservable",
      [](const std::string &type, PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        return qcor::createOperator(type, nativeHetMap);
      },
      "");

  m.def(
      "operatorTransform",
      [](const std::string &type, std::shared_ptr<Observable> obs) {
        return qcor::operatorTransform(type, obs);
      },
      "");
  m.def(
      "internal_observe",
      [](std::shared_ptr<CompositeInstruction> kernel,
         qcor::PauliOperator &obs) {
        auto q = ::qalloc(obs.nBits());
        return qcor::observe(kernel, obs, q);
      },
      "");
  m.def(
      "internal_observe",
      [](std::shared_ptr<CompositeInstruction> kernel, py::object obs) {
        auto observable = convertToQCOROperator(obs);
        auto q = ::qalloc(observable->nBits());
        return qcor::observe(kernel, observable, q);
      },
      "");

  m.def("internal_get_all_instructions", []() -> std::vector<py::tuple> {
    auto insts = xacc::getServices<xacc::Instruction>();
    std::vector<py::tuple> ret;
    for (auto inst : insts) {
      if (!inst->isComposite()) {
        ret.push_back(py::make_tuple(inst->name(), inst->nRequiredBits(),
                                     inst->isParameterized()));
      }
    }
    return ret;
  });

#ifdef QCOR_BUILD_MLIR_PYTHON_API
  m.def("openqasm_to_mlir",
        [](const std::string &oqasm_src, const std::string &kernel_name,
           bool add_entry_point) {
          return qcor::mlir_compile("openqasm", oqasm_src, kernel_name,
                                    qcor::OutputType::MLIR, add_entry_point);
        });

  m.def("openqasm_to_llvm_mlir", [](const std::string &oqasm_src,
                                    const std::string &kernel_name,
                                    bool add_entry_point) {
    return qcor::mlir_compile("openqasm", oqasm_src, kernel_name,
                              qcor::OutputType::LLVMMLIR, add_entry_point);
  });
  
  m.def("openqasm_to_llvm_ir",
        [](const std::string &oqasm_src, const std::string &kernel_name,
           bool add_entry_point) {
          return qcor::mlir_compile("openqasm", oqasm_src, kernel_name,
                                    qcor::OutputType::LLVMIR, add_entry_point);
        });
#endif 

  // QuaSiMo sub-module bindings:
  {
    py::module qsim =
        m.def_submodule("QuaSiMo", "QCOR's python QuaSiMo submodule");

    // QuantumSimulationModel bindings:
    py::class_<qcor::QuaSiMo::QuantumSimulationModel>(
        qsim, "QuantumSimulationModel",
        "The QuantumSimulationModel captures the quantum simulation problem "
        "description.")
        .def(py::init<>())
        .def(
            "__str__",
            [](qcor::QuaSiMo::QuantumSimulationModel &self) {
              std::stringstream ss;
              ss << "{ observable: " << self.observable->toString() << "}";
              return ss.str();
            },
            "");

    // ModelFactory bindings:
    py::class_<qcor::QuaSiMo::ModelFactory>(
        qsim, "ModelFactory",
        "The ModelFactory interface provides methods to "
        "construct QuaSiMo problem models.")
        .def(py::init<>())
        .def(
            "createModel",
            [](qcor::PauliOperator &obs, qcor::QuaSiMo::TdObservable ham_func) {
              return qcor::QuaSiMo::ModelFactory::createModel(obs, ham_func);
            },
            "Return the Model for a time-dependent problem.")
        .def(
            "createModel",
            [](py::object py_kernel, qcor::PauliOperator &obs,
               const int n_params) {
              qcor::QuaSiMo::QuantumSimulationModel model;
              auto nq = obs.nBits();
              auto kernel_functor = std::make_shared<qcor::PyKernelFunctor>(
                  py_kernel, nq, n_params);
              model.observable = &obs;
              model.user_defined_ansatz = kernel_functor;
              return std::move(model);
            },
            "")
        .def(
            "createModel",
            [](py::object py_kernel, py::object &py_obs, const int n_params) {
              qcor::QuaSiMo::QuantumSimulationModel model;
              static auto obs = convertToQCOROperator(py_obs);
              auto nq = obs->nBits();
              auto kernel_functor = std::make_shared<qcor::PyKernelFunctor>(
                  py_kernel, nq, n_params);
              model.observable = obs.get();
              model.user_defined_ansatz = kernel_functor;
              return std::move(model);
            },
            "")
        .def(
            "createModel",
            [](py::object py_kernel, qcor::PauliOperator &obs,
               const int n_qubits, const int n_params) {
              qcor::QuaSiMo::QuantumSimulationModel model;
              auto kernel_functor = std::make_shared<qcor::PyKernelFunctor>(
                  py_kernel, n_qubits, n_params);
              model.observable = &obs;
              model.user_defined_ansatz = kernel_functor;
              return std::move(model);
            },
            "")
        .def(
            "createModel",
            [](py::object py_kernel, std::shared_ptr<Observable> &obs,
               const int n_params) {
              qcor::QuaSiMo::QuantumSimulationModel model;
              auto nq = obs->nBits();
              auto kernel_functor = std::make_shared<qcor::PyKernelFunctor>(
                  py_kernel, nq, n_params);
              model.observable = obs.get();
              model.user_defined_ansatz = kernel_functor;
              return std::move(model);
            },
            "")

        .def(
            "createModel",
            [](py::object py_kernel, std::shared_ptr<Observable> &obs,
               const int n_qubits, const int n_params) {
              qcor::QuaSiMo::QuantumSimulationModel model;
              auto kernel_functor = std::make_shared<qcor::PyKernelFunctor>(
                  py_kernel, n_qubits, n_params);
              model.observable = obs.get();
              model.user_defined_ansatz = kernel_functor;
              return std::move(model);
            },
            "")
        .def(
            "createModel",
            [](qcor::PauliOperator &obs) {
              return qcor::QuaSiMo::ModelFactory::createModel(obs);
            },
            "")
        .def(
            "createModel",
            [](py::object &py_obs) {
              qcor::QuaSiMo::QuantumSimulationModel model;
              static auto obs = convertToQCOROperator(py_obs);
              model.observable = obs.get();
              return std::move(model);
            },
            "")
        .def(
            "createModel",
            [](qcor::QuaSiMo::ModelFactory::ModelType type,
               PyHeterogeneousMap &params) {
              auto nativeHetMap = heterogeneousMapConvert(params);
              return qcor::QuaSiMo::ModelFactory::createModel(type,
                                                              nativeHetMap);
            },
            "Create a model of a supported type.");
    py::enum_<qcor::QuaSiMo::ModelFactory::ModelType>(m, "ModelType")
        .value("Heisenberg", qcor::QuaSiMo::ModelFactory::ModelType::Heisenberg)
        .export_values();
    // CostFunctionEvaluator bindings
    py::class_<qcor::QuaSiMo::CostFunctionEvaluator,
               std::shared_ptr<qcor::QuaSiMo::CostFunctionEvaluator>,
               qcor::QuaSiMo::PyCostFunctionEvaluator>(
        qsim, "CostFunctionEvaluator",
        "The CostFunctionEvaluator interface provides methods to "
        "evaluate the observable operator expectation value on quantum "
        "backends.")
        .def(py::init<>())
        .def(
            "initialize",
            [](qcor::QuaSiMo::CostFunctionEvaluator &self,
               qcor::PauliOperator &obs) { return self.initialize(&obs); },
            "Initialize the evaluator")
        .def(
            "evaluate",
            [](qcor::QuaSiMo::CostFunctionEvaluator &self,
               std::shared_ptr<CompositeInstruction> state_prep) -> double {
              return self.evaluate(state_prep);
            },
            "Initialize the evaluator");
    qsim.def(
        "getObjEvaluator",
        [](qcor::PauliOperator &obs, const std::string &name = "default",
           py::dict p = {}) {
          return qcor::QuaSiMo::getObjEvaluator(obs, name);
        },
        py::arg("obs"), py::arg("name") = "default", py::arg("p") = py::dict(),
        py::return_value_policy::reference,
        "Return the CostFunctionEvaluator.");

    // QuantumSimulationWorkflow bindings
    py::class_<qcor::QuaSiMo::QuantumSimulationWorkflow,
               std::shared_ptr<qcor::QuaSiMo::QuantumSimulationWorkflow>,
               qcor::QuaSiMo::PyQuantumSimulationWorkflow>(
        qsim, "QuantumSimulationWorkflow",
        "The QuantumSimulationWorkflow interface provides methods to "
        "execute a quantum simulation workflow.")
        .def(py::init<>())
        .def(
            "execute",
            [](qcor::QuaSiMo::QuantumSimulationWorkflow &self,
               const qcor::QuaSiMo::QuantumSimulationModel &model)
                -> qcor::QuaSiMo::QuantumSimulationResult {
              return self.execute(model);
            },
            "Execute the workflow for the input problem model.");
    qsim.def(
        "getWorkflow",
        [](const std::string &name, PyHeterogeneousMap p = {}) {
          auto nativeHetMap = heterogeneousMapConvert(p);
          return qcor::QuaSiMo::getWorkflow(name, nativeHetMap);
        },
        py::arg("name"), py::arg("p") = PyHeterogeneousMap(),
        py::return_value_policy::reference,
        "Return the quantum simulation workflow.");
  }
}

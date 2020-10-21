#include "qcor/qcor_optimizer.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_pyqcor, m) {
  m.doc() = "Python bindings for QCOR.";
  // Expose QCOR API functions
  m.def(
      "createOptimizer",
      [](const std::string &name, py::dict p = {}) {
        xacc::HeterogeneousMap params;
        std::cout << "HOWDY: " << name << "\n";
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
  { py::module qsim = m.def_submodule("qsim", "QCOR's python qsim submodule"); }
}

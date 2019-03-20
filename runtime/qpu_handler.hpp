#ifndef RUNTIME_QPU_HANDLER_HPP_
#define RUNTIME_QPU_HANDLER_HPP_

#include <string>
#include "XACC.hpp"

namespace qcor {

class qpu_handler {

public:
  template <typename QuantumKernel>
  void vqe(QuantumKernel &&kernel, double observable, double optimizer) {
      xacc::info("qcor executing vqe.");
  }

  template <typename QuantumKernel> void execute(QuantumKernel &&kernel) {}

  template <typename QuantumKernel>
  void execute(const std::string &algorithm, QuantumKernel &&kernel) {}
};

} // namespace qcor

#endif
#ifndef RUNTIME_QPU_HANDLER_HPP_
#define RUNTIME_QPU_HANDLER_HPP_

#include "XACC.hpp"
#include "qcor.hpp"
#include "Function.hpp"
#include "AcceleratorBuffer.hpp"
#include "InstructionIterator.hpp"
#include <string>

namespace qcor {

class qpu_handler {

protected:

  std::shared_ptr<xacc::AcceleratorBuffer> buffer;

public:

  std::shared_ptr<xacc::AcceleratorBuffer> getResults() {
      return buffer;
  }

  template <typename QuantumKernel>
  void vqe(QuantumKernel &&kernel, double observable, double optimizer) {
    xacc::info("[qcor] Executing vqe! :)");

    
  }

  template <typename QuantumKernel> void execute(QuantumKernel &&kernel) {
    // xacc::info("[qcor] Executing circuit! :)");
    auto function = qcor::loadCompiledCircuit(kernel());

    int maxBitIdx = 0;
    xacc::InstructionIterator it(function);
      while (it.hasNext()) {
        auto nextInst = it.next();
        if (nextInst->isEnabled()) {
          for (auto& i : nextInst->bits()) {
              if (maxBitIdx < i) {
                  maxBitIdx = i;
              }
          }
        }
      }

    maxBitIdx++;

    // auto function = kernel();
    auto accelerator = xacc::getAccelerator();
    buffer = accelerator->createBuffer("q", maxBitIdx);
    accelerator->execute(buffer, function);
  }

  template <typename QuantumKernel>
  void execute(const std::string &algorithm, QuantumKernel &&kernel) {}
};

} // namespace qcor

#endif
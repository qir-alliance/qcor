#ifndef RUNTIME_QPU_HANDLER_HPP_
#define RUNTIME_QPU_HANDLER_HPP_

#include "qcor.hpp"

#include "AcceleratorBuffer.hpp"
#include "Function.hpp"
#include "XACC.hpp"

namespace qcor {

class qpu_handler {
protected:
  std::shared_ptr<xacc::AcceleratorBuffer> buffer;
public:
  std::shared_ptr<xacc::AcceleratorBuffer> getResults() { return buffer; }

  template <typename QuantumKernel>
  void vqe(QuantumKernel kernel, std::shared_ptr<Observable> observable,
           std::shared_ptr<Optimizer> optimizer) {
    auto function = qcor::loadCompiledCircuit(kernel());

    // std::cout << "Function:\n" << function->toString() << "\n";
    auto nPhysicalQubits = function->nPhysicalBits();
    auto accelerator = xacc::getAccelerator();
    buffer = accelerator->createBuffer("q", nPhysicalQubits);

    auto vqeAlgo = qcor::getAlgorithm("vqe");
    vqeAlgo->initialize(function, accelerator, buffer);
    vqeAlgo->execute(*observable.get(), *optimizer.get());
  }

  template <typename QuantumKernel> void execute(QuantumKernel &&kernel) {
    auto function = qcor::loadCompiledCircuit(kernel());
    auto nPhysicalQubits = function->nPhysicalBits();

    auto accelerator = xacc::getAccelerator();
    buffer = accelerator->createBuffer("q", nPhysicalQubits);
    accelerator->execute(buffer, function);
  }

  template <typename QuantumKernel>
  void execute(const std::string &algorithm, QuantumKernel &&kernel) {}
};

} // namespace qcor

#endif
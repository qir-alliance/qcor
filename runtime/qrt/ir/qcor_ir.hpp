#pragma once
#include <complex>
#include <functional>
#include <memory>
#include <vector>

#include "Identifiable.hpp"
#include "heterogeneous.hpp"
#include "qcor_pimpl.hpp"

// Leave Instruction opaque
namespace xacc {
class Instruction;
}

namespace qcor {
using LocalOpaqueInstPtr = std::shared_ptr<xacc::Instruction>;

class CompositeInstruction {
 private:
  class CompositeInstructionImpl;
  qcor_pimpl<CompositeInstructionImpl> m_internal;

 public:
  CompositeInstruction();
  ~CompositeInstruction();
  CompositeInstruction(std::shared_ptr<xacc::Identifiable> impl);
  CompositeInstruction(const std::string &&name);
  CompositeInstruction(const std::string &name);
  CompositeInstructionImpl *operator->();

  std::shared_ptr<xacc::Identifiable> get_as_opaque();

  const std::string name() const;
  int nInstructions();
  int nChildren();

  void setName(const std::string name);

  LocalOpaqueInstPtr getInstruction(const std::size_t idx);
  std::vector<LocalOpaqueInstPtr> getInstructions();
  void removeInstruction(const std::size_t idx);
  void replaceInstruction(const std::size_t idx, LocalOpaqueInstPtr newInst);
  void insertInstruction(const std::size_t idx, LocalOpaqueInstPtr newInst);
  void clear();

  void addInstruction(LocalOpaqueInstPtr instruction);
  void addInstructions(std::vector<LocalOpaqueInstPtr> &instruction);
  void addInstructions(const std::vector<LocalOpaqueInstPtr> &instruction);
  void addInstructions(const std::vector<LocalOpaqueInstPtr> &&insts,
                       bool shouldValidate = true);

  void attachMetadata(const int instId, xacc::HeterogeneousMap &&m);
  std::string toString();
};

}  // namespace qcor


/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include <memory>
#include <vector>

#include "Identifiable.hpp"
#include "heterogeneous.hpp"
#include "qcor_pimpl.hpp"

// Leave Instruction opaque
namespace xacc {
class Instruction;
class CompositeInstruction;
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

  std::size_t nLogicalBits();
  std::size_t nPhysicalBits();
  std::size_t nVariables();
  std::shared_ptr<CompositeInstruction> operator()(const std::vector<double>& x);

  std::shared_ptr<xacc::Identifiable> get_as_opaque();
  std::shared_ptr<xacc::CompositeInstruction> as_xacc();
  
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
  void addInstruction(std::shared_ptr<CompositeInstruction> composite);
  void addInstructions(std::vector<LocalOpaqueInstPtr> &instruction);
  void addInstructions(const std::vector<LocalOpaqueInstPtr> &instruction);
  void addInstructions(const std::vector<LocalOpaqueInstPtr> &&insts,
                       bool shouldValidate = true);

  void attachMetadata(const int instId, xacc::HeterogeneousMap &&m);
  std::string toString();
  int depth();
  
};

}  // namespace qcor


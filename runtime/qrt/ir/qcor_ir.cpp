/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "qcor_ir.hpp"

#include <cassert>

#include "Circuit.hpp"
#include "qcor_pimpl_impl.hpp"

namespace qcor {

/// ------------- CompositeInstruction Wrapper ---------------
class CompositeInstruction::CompositeInstructionImpl {
  friend class Operator;
  friend class CompositeInstruction;

 private:
  std::shared_ptr<xacc::CompositeInstruction> program;

 public:
  CompositeInstructionImpl() = default;
  CompositeInstructionImpl(const CompositeInstructionImpl &other)
      : program(other.program) {}
  CompositeInstructionImpl(std::shared_ptr<xacc::CompositeInstruction> comp)
      : program(comp) {}

  const std::string name() const { return program->name(); }
  const int nInstructions() { return program->nInstructions(); }
  const int nChildren() { return program->nChildren(); }

  void setName(const std::string name) { program->setName(name); }

  LocalOpaqueInstPtr getInstruction(const std::size_t idx) {
    return program->getInstruction(idx);
  }

  std::size_t nLogicalBits() { return program->nLogicalBits(); }
  std::size_t nPhysicalBits() { return program->nPhysicalBits(); }
  std::size_t nVariables() { return program->nVariables(); }

  std::vector<LocalOpaqueInstPtr> getInstructions() {
    auto tmp = program->getInstructions();
    std::vector<LocalOpaqueInstPtr> ret(tmp.begin(), tmp.end());
    return ret;
  }
  void removeInstruction(const std::size_t idx) {
    program->removeInstruction(idx);
  }
  void replaceInstruction(const std::size_t idx, LocalOpaqueInstPtr newInst) {
    program->replaceInstruction(
        idx, std::dynamic_pointer_cast<xacc::Instruction>(newInst));
  }
  void insertInstruction(const std::size_t idx, LocalOpaqueInstPtr newInst) {
    program->insertInstruction(
        idx, std::dynamic_pointer_cast<xacc::Instruction>(newInst));
  }
  void clear() { program->clear(); }

  std::shared_ptr<CompositeInstruction> operator()(
      const std::vector<double> &x) {
    return std::make_shared<CompositeInstruction>(program->operator()(x));
  }

  void addInstruction(LocalOpaqueInstPtr instruction) {
    program->addInstruction(
        std::dynamic_pointer_cast<xacc::Instruction>(instruction));
  }
  void addInstructions(std::vector<LocalOpaqueInstPtr> &instructions) {
    std::vector<xacc::InstPtr> add(instructions.begin(), instructions.end());
    program->addInstructions(add);
  }
  void addInstructions(const std::vector<LocalOpaqueInstPtr> &instructions) {
    addInstructions(instructions);
  }
  void addInstructions(const std::vector<LocalOpaqueInstPtr> &&insts,
                       bool shouldValidate = true) {
    program->addInstructions(
        std::vector<xacc::InstPtr>{insts.begin(), insts.end()}, shouldValidate);
  }
  void attachMetadata(const int instId, xacc::HeterogeneousMap &&m) {
    program->getInstruction(instId)->attachMetadata(std::move(m));
  }
  std::string toString() { return program->toString(); }
};

CompositeInstruction::CompositeInstructionImpl *
CompositeInstruction::operator->() {
  return m_internal.operator->();
}

CompositeInstruction::CompositeInstruction(
    std::shared_ptr<xacc::Identifiable> impl)
    : m_internal(std::dynamic_pointer_cast<xacc::CompositeInstruction>(impl)) {}
CompositeInstruction::CompositeInstruction() = default;
CompositeInstruction::~CompositeInstruction() = default;
CompositeInstruction::CompositeInstruction(const std::string &&name)
    : m_internal(std::make_shared<xacc::quantum::Circuit>(name)) {}
CompositeInstruction::CompositeInstruction(const std::string &name)
    : m_internal(std::make_shared<xacc::quantum::Circuit>(name)) {}

std::shared_ptr<xacc::Identifiable> CompositeInstruction::get_as_opaque() {
  return m_internal->program;
}

const std::string CompositeInstruction::name() const {
  return m_internal->name();
}
std::shared_ptr<xacc::CompositeInstruction> CompositeInstruction::as_xacc() {
  return m_internal->program;
}
void CompositeInstruction::addInstruction(
    std::shared_ptr<CompositeInstruction> composite) {
  m_internal->program->addInstruction(composite->as_xacc());
}

int CompositeInstruction::depth() { return m_internal->program->depth(); }

std::size_t CompositeInstruction::nLogicalBits() {
  return m_internal->nLogicalBits();
}
std::size_t CompositeInstruction::nPhysicalBits() {
  return m_internal->nPhysicalBits();
}
std::size_t CompositeInstruction::nVariables() {
  return m_internal->nVariables();
}

std::shared_ptr<CompositeInstruction> CompositeInstruction::operator()(
    const std::vector<double> &x) {
  return m_internal->operator()(x);
}

std::string CompositeInstruction::toString() { return m_internal->toString(); }

void CompositeInstruction::attachMetadata(const int instId,
                                          xacc::HeterogeneousMap &&m) {
  m_internal->attachMetadata(instId, std::move(m));
}

int CompositeInstruction::nInstructions() {
  return m_internal->nInstructions();
}
int CompositeInstruction::nChildren() { return m_internal->nChildren(); }

void CompositeInstruction::setName(const std::string name) {
  m_internal->setName(name);
}

LocalOpaqueInstPtr CompositeInstruction::getInstruction(const std::size_t idx) {
  return m_internal->getInstruction(idx);
}

std::vector<LocalOpaqueInstPtr> CompositeInstruction::getInstructions() {
  return m_internal->getInstructions();
}
void CompositeInstruction::removeInstruction(const std::size_t idx) {
  m_internal->removeInstruction(idx);
}
void CompositeInstruction::replaceInstruction(const std::size_t idx,
                                              LocalOpaqueInstPtr newInst) {
  m_internal->replaceInstruction(idx, newInst);
}
void CompositeInstruction::insertInstruction(const std::size_t idx,
                                             LocalOpaqueInstPtr newInst) {
  m_internal->insertInstruction(idx, newInst);
}
void CompositeInstruction::clear() { m_internal->clear(); }

void CompositeInstruction::addInstruction(LocalOpaqueInstPtr instruction) {
  m_internal->addInstruction(instruction);
}
void CompositeInstruction::addInstructions(
    std::vector<LocalOpaqueInstPtr> &instructions) {
  m_internal->addInstructions(instructions);
}
void CompositeInstruction::addInstructions(
    const std::vector<LocalOpaqueInstPtr> &instructions) {
  addInstructions(instructions);
}
void CompositeInstruction::addInstructions(
    const std::vector<LocalOpaqueInstPtr> &&insts, bool shouldValidate) {
  m_internal->addInstructions(std::move(insts), shouldValidate);
}

}  // namespace qcor
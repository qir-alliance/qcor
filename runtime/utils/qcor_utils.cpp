#include "qcor_utils.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

namespace qcor {
void set_verbose(bool verbose) { xacc::set_verbose(verbose); }
bool get_verbose() { return xacc::verbose; }
void set_shots(const int shots) { ::quantum::set_shots(shots); }
void error(const std::string &msg) { xacc::error(msg); }
ResultsBuffer sync(Handle &handle) { return handle.get(); }

void set_backend(const std::string &backend) {
  xacc::internal_compiler::compiler_InitializeXACC();
  xacc::internal_compiler::setAccelerator(backend.c_str());
}

std::shared_ptr<xacc::CompositeInstruction> compile(const std::string &src) {
  return xacc::getCompiler("xasm")->compile(src)->getComposites()[0];
}

namespace __internal__ {
std::shared_ptr<qcor::CompositeInstruction> create_composite(std::string name) {
  return xacc::getIRProvider("quantum")->createComposite(name);
}
std::shared_ptr<qcor::CompositeInstruction> create_ctrl_u() {
  return std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      xacc::getService<xacc::Instruction>("C-U"));
}
std::shared_ptr<qcor::IRTransformation>
get_transformation(const std::string &transform_type) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getService<xacc::IRTransformation>(transform_type);
}
std::shared_ptr<qcor::CompositeInstruction>
decompose_unitary(const std::string algorithm, UnitaryMatrix &mat, const std::string buffer_name) {
  auto tmp = xacc::getService<xacc::Instruction>(algorithm);
  auto decomposed = std::dynamic_pointer_cast<CompositeInstruction>(tmp);

  // default Adam
  auto optimizer = xacc::getOptimizer("mlpack");
  const bool expandOk = decomposed->expand(
      {std::make_pair("unitary", mat), std::make_pair("optimizer", optimizer)});

  if (!expandOk) {
    xacc::error("Could not decmpose unitary with " + algorithm);
  }

  for (auto inst : decomposed->getInstructions()) {
      std::vector<std::string> buffer_names;
      for (int i = 0; i < inst->nRequiredBits(); i++) {
          buffer_names.push_back(buffer_name);
      }
      inst->setBufferNames(buffer_names);
  }

  return decomposed;
}
} // namespace __internal__
} // namespace qcor
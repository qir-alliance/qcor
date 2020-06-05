#include "xacc_llvm_compiler.hpp"
#include "Circuit.hpp"
#include "qcor_clang_wrapper.hpp"
#include "qrt_mapper.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include "llvm/IR/Constants.h"
#include <clang/CodeGen/CodeGenAction.h>
#include <cxxabi.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

using namespace cppmicroservices;

using namespace xacc::quantum;

namespace qcor {

std::shared_ptr<xacc::IR>
LLVMCompiler::compile(const std::string &src,
                      std::shared_ptr<Accelerator> acc) {
  return nullptr;
}

const std::string
LLVMCompiler::translate(std::shared_ptr<CompositeInstruction> program,
                        HeterogeneousMap &extra_data) {

  auto bufferNames = extra_data.get<std::vector<std::string>>("buffer-names");
  auto qpu_name = extra_data.getString("accelerator");
  auto func_proto = extra_data.getString("function-prototype");
  auto kernel_name = extra_data.getString("kernel-name");

  auto visitor = std::make_shared<qrt_mapper>(program->name());
  xacc::InstructionIterator iter(program);
  while (iter.hasNext()) {
    auto next = iter.next();
    next->accept(visitor);
  }
  auto src = visitor->get_new_src();

  std::stringstream ss;

  ss << "#include \"qrt.hpp\"\n";
  ss << "#include \"xacc_internal_compiler.hpp\"\n";
  ss << "__attribute__((annotate(\"quantum\"))) " << func_proto << " {\n";
  ss << "quantum::initialize(\"" << qpu_name << "\", \"" << kernel_name
     << "\");\n";
  for (auto &buf : bufferNames) {
    ss << buf << ".setNameAndStore(\"" + buf + "\");\n";
  }

  ss << src;
  ss << "if (__execute) {\n";

  if (bufferNames.size() > 1) {
    ss << "xacc::AcceleratorBuffer * buffers[" << bufferNames.size() << "] = {";
    ss << bufferNames[0] << ".results()";
    for (unsigned int k = 1; k < bufferNames.size(); k++) {
      ss << ", " << bufferNames[k] << ".results()";
    }
    ss << "};\n";
    ss << "quantum::submit(buffers," << bufferNames.size();
  } else {
    ss << "quantum::submit(" << bufferNames[0] << ".results()";
  }
  ss << ");\n}\n}";

//   llvm::errs() << "code is \n" << ss.str() << "\n";
  auto act = qcor::emit_llvm_ir(ss.str());
  auto module = act->takeModule();

  std::string ret_str2;
  llvm::raw_string_ostream rso2(ret_str2);
  module->print(rso2, nullptr);
  rso2.flush();

  return ret_str2;
}

} // namespace qcor

namespace {

/**
 */
class US_ABI_LOCAL LLVMCompilerActivator : public BundleActivator {

public:
  LLVMCompilerActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::LLVMCompiler>();
    context.RegisterService<xacc::Compiler>(xt);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(LLVMCompilerActivator)

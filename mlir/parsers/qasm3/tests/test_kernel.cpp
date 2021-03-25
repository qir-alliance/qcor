#include "clang/CodeGen/CodeGenAction.h"
#include "gtest/gtest.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ToolOutputFile.h"
#include "qcor_clang_wrapper.hpp"
#include "qcor_mlir_api.hpp"
#include "qcor_syntax_handler.hpp"

TEST(qasm3VisitorTester, checkGlobalConstInSubroutine) {
  const std::string kernel_test = R"#(OPENQASM 3;
include "qelib1.inc";
int i = 10;
kernel test_this(int) -> int;
int j = test_this(i);
print(j);
QCOR_EXPECT_TRUE(j == 20);
)#";

  auto mlir = qcor::mlir_compile("qasm3", kernel_test, "kernel_test",
                                 qcor::OutputType::MLIR, true);
  std::cout << mlir << "\n";

  auto llvm = qcor::mlir_compile("qasm3", kernel_test, "kernel_test",
                                 qcor::OutputType::LLVMIR, true);
  std::cout << llvm << "\n";

  // -------------------------------------------//
  // Create an external llvm module containing the
  // actual kernel function code...
  qcor::__internal__developer__flags__::add_predefines = false;  // Don't let QCOR SH AddPredefines run
  auto act = qcor::emit_llvm_ir(R"#(extern "C" {
int test_this(int i) { return i + 10; }
}
)#");
  auto module = act->takeModule();

  // Add the module to a vector to pass to the JIT execute function
  std::vector<std::unique_ptr<llvm::Module>> extra_code_to_link;
  extra_code_to_link.push_back(std::move(module));
  // -------------------------------------------//

  EXPECT_FALSE(qcor::execute("qasm3", kernel_test, "kernel_test", extra_code_to_link, 0));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}

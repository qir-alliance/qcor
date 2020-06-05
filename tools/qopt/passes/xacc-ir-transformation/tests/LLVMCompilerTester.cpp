#include "xacc.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>
#include <heterogeneous.hpp>
using namespace xacc;

TEST(LLVMCompilerTester, checkSimple) {

  auto f = xacc::getService<xacc::Compiler>("xasm")
               ->compile(R"#(__qpu__ void f(qreg q) {H(q[0]);})#")
               ->getComposite("f");

  auto llvm_compiler = xacc::getService<xacc::Compiler>("xacc-llvm");
  HeterogeneousMap extra_data{
      std::make_pair("accelerator", "qpp"), std::make_pair("kernel-name", "f"),
      std::make_pair("buffer-names", std::vector<std::string>{"q"}),
      std::make_pair("function-prototype", "void f(qreg q)")};
  auto translated = llvm_compiler->translate(f, extra_data);

  std::cout << "translated:\n" << translated << "\n";

}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}

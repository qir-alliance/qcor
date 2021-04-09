#include <xacc.hpp>

#include "clang/Sema/DeclSpec.h"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "token_collector.hpp"
#include "xacc_service.hpp"
#include "qcor_config.hpp"
#include "xacc_config.hpp"

TEST(PyXASMTokenCollectorTester, checkSimple) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"(
    H(qb[0])
    CX(qb[0],qb[1])
    for i in range(qb.size()):
        X(qb[i])
        X(qb[i])
        Measure(qb[i])
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"qb"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";

  EXPECT_EQ(R"#(quantum::h(qb[0]);
quantum::cnot(qb[0], qb[1]);
for (auto i : range(qb.size())) {
quantum::x(qb[i]);
quantum::x(qb[i]);
quantum::mz(qb[i]);
}
)#",
            ss.str());
}

TEST(PyXASMTokenCollectorTester, checkIf) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"(
    H(qb[0])
    CX(qb[0],qb[1])
    for i in range(qb.size()):
      if Measure(qb[i]):
        X(qb[i])
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"qb"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(quantum::h(qb[0]);
quantum::cnot(qb[0], qb[1]);
for (auto i : range(qb.size())) {
if (quantum::mz(qb[i])) {
quantum::x(qb[i]);
}
}
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

TEST(PyXASMTokenCollectorTester, checkPythonList) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"(
    # inline initializer list
    apply_X_at_idx.ctrl([q[1], q[2]], q[0])
    # array var assignement
    array_val = [q[1], q[2]]
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"qb"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(apply_X_at_idx::ctrl(parent_kernel, {q[1], q[2]}, q[0]);
auto array_val = {q[1], q[2]}; 
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

int main(int argc, char **argv) {
  std::string xacc_config_install_dir = std::string(XACC_INSTALL_DIR);
  std::string qcor_root = std::string(QCOR_INSTALL_DIR);
  if (xacc_config_install_dir != qcor_root) {
    xacc::addPluginSearchPath(std::string(QCOR_INSTALL_DIR) + "/plugins");
  }
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}

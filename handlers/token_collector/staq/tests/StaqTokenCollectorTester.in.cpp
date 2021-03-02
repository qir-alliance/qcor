#include <xacc.hpp>

#include "clang/Sema/DeclSpec.h"
#include "gtest/gtest.h"
#include "qalloc.hpp"
#include "qcor_config.hpp"
#include "test_utils.hpp"
#include "token_collector.hpp"
#include "xacc_config.hpp"
#include "xacc_service.hpp"

TEST(StaqTokenCollectorTester, checkSimple) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(
      R"#(oracle adder a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3 { "@CMAKE_BINARY_DIR@/handlers/token_collector/staq/tests/add_3_5.v" }

  creg result[4];
  // a = 3
  x a[0];
  x a[1];

  // b = 5
  x b[0];
  x b[2];

  adder a[0],a[1],a[2],a[3],b[0],b[1],b[2],b[3],c[0],c[1],c[2],c[3];

  measure c[2] -> result[2];

  // measure
  measure c -> result;)#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("staq");
  xasm_tc->collect(*PP.get(), cached, {"a", "b", "c"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";

  EXPECT_EQ(R"#(auto anc = qalloc(2147483647);
quantum::x(a[0]);
quantum::x(a[1]);
quantum::x(b[0]);
quantum::x(b[2]);
quantum::cnot(a[0], c[0]);
quantum::cnot(b[0], c[0]);
quantum::h(anc[0]);
quantum::cnot(b[0], anc[0]);
quantum::tdg(anc[0]);
quantum::cnot(a[0], anc[0]);
quantum::t(anc[0]);
quantum::cnot(b[0], anc[0]);
quantum::tdg(anc[0]);
quantum::cnot(a[0], anc[0]);
quantum::t(anc[0]);
quantum::cnot(a[0], b[0]);
quantum::tdg(b[0]);
quantum::cnot(a[0], b[0]);
quantum::t(a[0]);
quantum::t(b[0]);
quantum::h(anc[0]);
quantum::cnot(a[1], c[1]);
quantum::cnot(b[1], c[1]);
quantum::cnot(anc[0], c[1]);
quantum::h(anc[1]);
quantum::cnot(b[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(b[1], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], b[1]);
quantum::tdg(b[1]);
quantum::cnot(a[1], b[1]);
quantum::t(a[1]);
quantum::tdg(b[1]);
quantum::cnot(anc[0], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(anc[0], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(a[1], anc[0]);
quantum::tdg(anc[0]);
quantum::cnot(a[1], anc[0]);
quantum::t(a[1]);
quantum::t(anc[0]);
quantum::cnot(b[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(anc[0], anc[1]);
quantum::t(anc[1]);
quantum::cnot(b[1], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(anc[0], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(anc[0], b[1]);
quantum::tdg(b[1]);
quantum::cnot(anc[0], b[1]);
quantum::t(anc[0]);
quantum::tdg(b[1]);
quantum::h(anc[1]);
quantum::cnot(a[2], c[2]);
quantum::cnot(b[2], c[2]);
quantum::cnot(anc[1], c[2]);
quantum::h(anc[2]);
quantum::cnot(b[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(b[2], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], b[2]);
quantum::tdg(b[2]);
quantum::cnot(a[2], b[2]);
quantum::t(a[2]);
quantum::tdg(b[2]);
quantum::cnot(anc[1], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(anc[1], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(a[2], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[2], anc[1]);
quantum::t(a[2]);
quantum::t(anc[1]);
quantum::cnot(b[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(anc[1], anc[2]);
quantum::t(anc[2]);
quantum::cnot(b[2], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(anc[1], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(anc[1], b[2]);
quantum::tdg(b[2]);
quantum::cnot(anc[1], b[2]);
quantum::t(anc[1]);
quantum::tdg(b[2]);
quantum::h(anc[2]);
quantum::cnot(a[3], c[3]);
quantum::cnot(b[3], c[3]);
quantum::cnot(anc[2], c[3]);
quantum::h(anc[2]);
quantum::cnot(b[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(b[2], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], b[2]);
quantum::tdg(b[2]);
quantum::cnot(a[2], b[2]);
quantum::t(a[2]);
quantum::tdg(b[2]);
quantum::cnot(anc[1], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(anc[1], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(a[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(a[2], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[2], anc[1]);
quantum::t(a[2]);
quantum::t(anc[1]);
quantum::cnot(b[2], anc[2]);
quantum::t(anc[2]);
quantum::cnot(anc[1], anc[2]);
quantum::t(anc[2]);
quantum::cnot(b[2], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(anc[1], anc[2]);
quantum::tdg(anc[2]);
quantum::cnot(anc[1], b[2]);
quantum::tdg(b[2]);
quantum::cnot(anc[1], b[2]);
quantum::t(anc[1]);
quantum::tdg(b[2]);
quantum::h(anc[2]);
quantum::h(anc[1]);
quantum::cnot(b[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(b[1], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], b[1]);
quantum::tdg(b[1]);
quantum::cnot(a[1], b[1]);
quantum::t(a[1]);
quantum::tdg(b[1]);
quantum::cnot(anc[0], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(anc[0], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(a[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(a[1], anc[0]);
quantum::tdg(anc[0]);
quantum::cnot(a[1], anc[0]);
quantum::t(a[1]);
quantum::t(anc[0]);
quantum::cnot(b[1], anc[1]);
quantum::t(anc[1]);
quantum::cnot(anc[0], anc[1]);
quantum::t(anc[1]);
quantum::cnot(b[1], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(anc[0], anc[1]);
quantum::tdg(anc[1]);
quantum::cnot(anc[0], b[1]);
quantum::tdg(b[1]);
quantum::cnot(anc[0], b[1]);
quantum::t(anc[0]);
quantum::tdg(b[1]);
quantum::h(anc[1]);
quantum::h(anc[0]);
quantum::cnot(b[0], anc[0]);
quantum::tdg(anc[0]);
quantum::cnot(a[0], anc[0]);
quantum::t(anc[0]);
quantum::cnot(b[0], anc[0]);
quantum::tdg(anc[0]);
quantum::cnot(a[0], anc[0]);
quantum::t(anc[0]);
quantum::cnot(a[0], b[0]);
quantum::tdg(b[0]);
quantum::cnot(a[0], b[0]);
quantum::t(a[0]);
quantum::t(b[0]);
quantum::h(anc[0]);
quantum::mz(c[2]);
quantum::mz(c[0]);
quantum::mz(c[1]);
quantum::mz(c[2]);
quantum::mz(c[3]);
)#",
            ss.str());
}

TEST(StaqTokenCollectorTester, checkReset) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(
creg c[1];
h q[0];
reset q[0];
reset q[1];
x q[0];
measure q[0] -> c[0];)#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("staq");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  EXPECT_TRUE(ss.str().find("quantum::reset(q[0]);") != std::string::npos);
  EXPECT_TRUE(ss.str().find("quantum::reset(q[1]);") != std::string::npos);
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

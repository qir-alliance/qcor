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
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(apply_X_at_idx::ctrl(parent_kernel, {q[1], q[2]}, q[0]);
auto array_val = {q[1], q[2]}; 
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

TEST(PyXASMTokenCollectorTester, checkStringLiteral) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"(
    # Cpp style strings
    print("hello", 1, "world")
    # Python style
    print('howdy', 1, 'abc')
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(print("hello", 1, "world");
print("howdy", 1, "abc");
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

TEST(PyXASMTokenCollectorTester, checkQregMethods) {
  LexerHelper helper;
  auto [tokens, PP] = helper.Lex(R"(
    ctrl_qubits = q.head(q.size()-1)
    last_qubit = q.tail()
    Z.ctrl(ctrl_qubits, last_qubit)
    
    # inline
    X.ctrl(q.head(q.size()-1), q.tail())

    # range:
    # API
    r = q.extract_range(0, bitPrecision)
    # Python style
    slice1 = q[0:3]
    # step size
    slice2 = q[0:5:2]
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(auto ctrl_qubits = q.head(q.size()-1); 
auto last_qubit = q.tail(); 
Z::ctrl(parent_kernel, ctrl_qubits, last_qubit);
X::ctrl(parent_kernel, q.head(q.size()-1), q.tail());
auto r = q.extract_range(0,bitPrecision); 
auto slice1 = q.extract_range({static_cast<size_t>(0), static_cast<size_t>(3)}); 
auto slice2 = q.extract_range({static_cast<size_t>(0), static_cast<size_t>(5), static_cast<size_t>(2)}); 
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

TEST(PyXASMTokenCollectorTester, checkBroadCastWithSlice) {
  LexerHelper helper;
  auto [tokens, PP] = helper.Lex(R"(
    X(q.head(q.size()-1))
    X(q[0])
    X(q)
    X(q[0:2])
    X(q[0:5:2])
    Measure(q.head(q.size()-1))
    Measure(q[0])
    Measure(q)
    Measure(q[0:2])
    Measure(q[0:5:2])
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(quantum::x(q.head(q.size()-1));
quantum::x(q[0]);
quantum::x(q);
quantum::x(q.extract_range({static_cast<size_t>(0), static_cast<size_t>(2)}));
quantum::x(q.extract_range({static_cast<size_t>(0), static_cast<size_t>(5), static_cast<size_t>(2)}));
quantum::mz(q.head(q.size()-1));
quantum::mz(q[0]);
quantum::mz(q);
quantum::mz(q.extract_range({static_cast<size_t>(0), static_cast<size_t>(2)}));
quantum::mz(q.extract_range({static_cast<size_t>(0), static_cast<size_t>(5), static_cast<size_t>(2)}));
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

TEST(PyXASMTokenCollectorTester, checkQcorOperators) {
  LexerHelper helper;
  auto [tokens, PP] = helper.Lex(R"(
    exponent_op = X(0) * Y(1) - Y(0) * X(1)
    exp_i_theta(q, theta, exponent_op)
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(auto exponent_op = X(0)*Y(1)-Y(0)*X(1); 
quantum::exp(q, theta, exponent_op);
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

TEST(PyXASMTokenCollectorTester, checkCommonMath) {
  LexerHelper helper;
  auto [tokens, PP] = helper.Lex(R"(
    out_parity = oneCount - 2 * (oneCount / 2)
    # Power
    index = 2**n 
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {""}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string expectedCodeGen =
      R"#(auto out_parity = oneCount-2*(oneCount/2); 
auto index = std::pow(2, n); 
)#";
  EXPECT_EQ(expectedCodeGen, ss.str());
}

TEST(PyXASMTokenCollectorTester, checkKernelSignature) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"(
    # fake local var creation 
    # (should be from the function args if compiling full source.)
    callable = createCallable(a,b,c)
    callable.ctrl([q[1], q[2]], q[0])
    callable.adjoint(q)
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  const std::string code_gen_str = ss.str();
  // Rewrite to '.'
  EXPECT_TRUE(code_gen_str.find("callable.ctrl(parent_kernel, {q[1], q[2]}, q[0]);") != std::string::npos);
  EXPECT_TRUE(code_gen_str.find("callable.adjoint(parent_kernel, q);") != std::string::npos);
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

#include "test_utils.hpp"
#include "token_collector_util.hpp"
#include "xacc_service.hpp"
#include "clang/Sema/DeclSpec.h"
#include "gtest/gtest.h"
#include <xacc.hpp>

TEST(TokenCollectorTester, checkSimple) {

  LexerHelper helper;

  auto [tokens, PP] =
      helper.Lex("H(q[0]);\nCX(q[0],q[1]);\nRy(q[3], theta);\nRx(q[0], "
                 "2.2);\nfor (int i = 0; i < "
                 "q.size(); i++) {\n  "
                 "Measure(q[i]);\n}\nunknown_func_let_compiler_find_it(q, 0, "
                 "q.size(), 0);\n");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  auto results = qcor::run_token_collector(*PP, cached, {"q"});

  std::cout << results << "\n";
}

TEST(TokenCollectorTester, checkQPE) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(const auto nQubits = q.size();
  // Last qubit is the eigenstate of the unitary operator 
  // hence, prepare it in |1> state
  X(q[nQubits - 1]);

  // Apply Hadamard gates to the counting qubits:
  for (int qIdx = 0; qIdx < nQubits - 1; ++qIdx) {
    H(q[qIdx]);
  }

  // Apply Controlled-Oracle: in this example, Oracle is T gate;
  // i.e. Ctrl(T) = CPhase(pi/4)
  const auto bitPrecision = nQubits - 1;
  for (int32_t i = 0; i < bitPrecision; ++i) {
    const int nbCalls = 1 << i;
    for (int j = 0; j < nbCalls; ++j) {
      int ctlBit = i;
      // Controlled-Oracle
      Controlled::Apply(ctlBit, compositeOp, q);
    }
  }

  // Inverse QFT on the counting qubits:
  int startIdx = 0;
  int shouldSwap = 1;
  iqft(q, startIdx, bitPrecision, shouldSwap);

  // Measure counting qubits
  for (int qIdx = 0; qIdx < bitPrecision; ++qIdx) {
    Measure(q[qIdx]);
  })#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }
  auto results =
      qcor::run_token_collector(*PP, cached, {"q"});
  std::cout << results << "\n";
}

TEST(TokenCollectorTester, checkOpenQasm) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(using qcor::openqasm;
  h r[0];
  cx r[0], r[1];
  creg c[2];
  measure r -> c;
  )#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }
  auto results =
      qcor::run_token_collector(*PP, cached, {"r"});
  std::cout << results << "\n";

  EXPECT_EQ(R"#(quantum::h(r[0]);
quantum::cnot(r[0], r[1]);
quantum::mz(r[0]);
quantum::mz(r[1]);
)#", results);
}


TEST(TokenCollectorTester, checkMixed) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(
  H(q[0]);
  CX(q[0], q[1]);

  using qcor::openqasm;

  h r[0];
  cx r[0], r[1];

  using qcor::xasm;
  
  for (int i = 0; i < q.size(); i++) {
    Measure(q[i]);
    Measure(r[i]);
  }
  )#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }
  auto results =
      qcor::run_token_collector(*PP, cached, {"r"});
  std::cout << results << "\n";

  EXPECT_EQ(R"#(quantum::h(q[0]);
quantum::cnot(q[0], q[1]);
quantum::h(r[0]);
quantum::cnot(r[0], r[1]);
for ( int i = 0 ; i < q.size() ; i ++ ) { 
quantum::mz(q[i]);
quantum::mz(r[i]);
} 
)#", results);
}
int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}

#include "clang/Sema/DeclSpec.h"
#include "gtest/gtest.h"
#include "qcor_config.hpp"
#include "test_utils.hpp"
#include "token_collector.hpp"
#include "xacc.hpp"
#include "xacc_config.hpp"
#include "xacc_service.hpp"

TEST(XASMTokenCollectorTester, checkSimple) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(
      "H(q[0]);\nCX(q[0],q[1]);\nRy(q[3], theta);\nRx(q[0], "
      "2.2);\nfor (int i = 0; i < "
      "q.size(); i++) {\n  Measure(q[i]);\n}\nexp_i_theta(q, theta, "
      "observable);\n");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("xasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";

  EXPECT_EQ(R"#(quantum::h(q[0]);
quantum::cnot(q[0], q[1]);
quantum::ry(q[3], theta);
quantum::rx(q[0], 2.2);
for ( int i = 0 ; i < q.size() ; i ++ ) { 
quantum::mz(q[i]);
} 
quantum::exp(q, theta, observable);
)#",
            ss.str());
}

TEST(XASMTokenCollectorTester, checkComplexQaoa) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(// Local Declarations
  auto nQubits = q.size();
  int gamma_counter = 0;
  int beta_counter = 0;

  // Start of in the uniform superposition
  for (int i = 0; i < nQubits; i++) {
    H(q[0]);
  }

  // Get all non-identity hamiltonian terms
  // for the following exp(H_i) trotterization
  auto cost_terms = cost_ham.getNonIdentitySubTerms();

  // Loop over qaoa steps
  for (int step = 0; step < n_steps; step++) {

    // Loop over cost hamiltonian terms
    for (int i = 0; i < cost_terms.size(); i++) {

      // for xasm we have to allocate the variables
      // cant just run exp_i_theta(q, gamma[gamma_counter], cost_terms[i]) yet
      // :(
      auto cost_term = cost_terms[i];
      auto m_gamma = gamma[gamma_counter];

      // trotterize
      exp_i_theta(q, m_gamma, cost_term);

      gamma_counter++;
    }

    // Add the reference hamiltonian term
    for (int i = 0; i < nQubits; i++) {
      auto ref_ham_term = qcor::X(i);
      auto m_beta = beta[beta_counter];
      exp_i_theta(q, m_beta, ref_ham_term);
      beta_counter++;
    }
  })#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("xasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << ss.str() << "\n";
  EXPECT_EQ(R"#(auto nQubits = q.size() ; 
int gamma_counter = 0 ; 
int beta_counter = 0 ; 
for ( int i = 0 ; i < nQubits ; i ++ ) { 
quantum::h(q[0]);
} 
auto cost_terms = cost_ham.getNonIdentitySubTerms() ; 
for ( int step = 0 ; step < n_steps ; step ++ ) { 
for ( int i = 0 ; i < cost_terms.size() ; i ++ ) { 
auto cost_term = cost_terms[i] ; 
auto m_gamma = gamma[gamma_counter] ; 
quantum::exp(q, m_gamma, cost_term);
gamma_counter ++ ; 
} 
for ( int i = 0 ; i < nQubits ; i ++ ) { 
auto ref_ham_term = qcor::X(i) ; 
auto m_beta = beta[beta_counter] ; 
quantum::exp(q, m_beta, ref_ham_term);
beta_counter ++ ; 
} 
} 
)#",
            ss.str());
}

TEST(XASMTokenCollectorTester, checkReset) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex("H(q[0]);\nReset(q[0]);\nRy(q[3], theta);");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("xasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";
  EXPECT_TRUE(ss.str().find("quantum::reset(q[0]);") != std::string::npos);
}

TEST(XASMTokenCollectorTester, checkCharLiterals) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(
      "if (c == '1') {\nCNOT(q[0],q[1]);\n}\n"
      "std::cout << \"wow a single quote: \" << '\\'';\n");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("xasm");
  xasm_tc->collect(*PP.get(), cached, {"q"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";

  EXPECT_EQ(R"#(if ( c == '1' ) { 
quantum::cnot(q[0], q[1]);
} 
std :: cout << "wow a single quote: " << '\'' ; 
)#",
            ss.str());
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

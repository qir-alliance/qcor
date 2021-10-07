/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "test_utils.hpp"
#include "token_collector.hpp"
#include "xacc_service.hpp"
#include "clang/Sema/DeclSpec.h"
#include "gtest/gtest.h"
#include <xacc.hpp>
#include "qalloc.hpp"
#include "xacc_config.hpp"
#include "qcor_config.hpp"

TEST(UnitaryTokenCollectorTester, checkSimple) {
  
  LexerHelper helper;

  auto [tokens, PP] =
      helper.Lex(R"#(using qcor::unitary;
  auto ccnot = UnitaryMatrix::Identity(8, 8);
  ccnot(6, 6) = 0.0;
  ccnot(7, 7) = 0.0;
  ccnot(6, 7) = 1.0;
  ccnot(7, 6) = 1.0;)#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("unitary");
  xasm_tc->collect(*PP.get(), cached, {"a"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";

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

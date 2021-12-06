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
#include "unitary_token_collector.hpp"

#include "InstructionIterator.hpp"
#include "qalloc.hpp"
#include "qrt_mapper.hpp"
#include "xacc.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <Utils.hpp>
#include <iostream>
#include <memory>
#include <set>

#include "clang/Basic/TokenKinds.h"
#include "clang/Parse/Parser.h"

using namespace cppmicroservices;
using namespace clang;

namespace {

/**
 */
class US_ABI_LOCAL UnitaryTokenCollectorActivator : public BundleActivator {

public:
  UnitaryTokenCollectorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto st = std::make_shared<qcor::UnitaryTokenCollector>();
    context.RegisterService<qcor::TokenCollector>(st);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(UnitaryTokenCollectorActivator)

namespace qcor {

void UnitaryTokenCollector::collect(clang::Preprocessor &PP,
                                    clang::CachedTokens &Toks,
                                    std::vector<std::string> bufferNames,
                                    std::stringstream &ss, const std::string &kernel_name) {

  // First figure out the variable name of the UnitaryMatrix
  // Next, take all the tokens and just add them to the
  // string stream (we want to keep the code as is).
  // Finally, set up the call to QFAST to decompose the matrix
  // to qasm, set that as the QRT program and hook up submission
  // to the backend

  // To get the var name, find first occurrence of UnitaryMatrix
  // and check if previous token is an equal. If it is, token before that
  // is the var name, if not, the token is the next one

  std::string var_name = "";
  for (int i = 0; i < Toks.size(); i++) {
    auto current_token_str = PP.getSpelling(Toks[i]);
    if (var_name.empty()) {
      if (current_token_str == "UnitaryMatrix") {
        auto next = PP.getSpelling(Toks[i + 1]);
        if (next == "::") {
          // this means we are on the rhs of equal sign
          var_name = PP.getSpelling(Toks[i - 2]);
        } else if (next == ":" && PP.getSpelling(Toks[i+2]) == ":") {
          var_name = PP.getSpelling(Toks[i - 2]);
        } else {
          // this means we are on lhs of equal sign
          var_name = next;
        }
      }
    }

    if (current_token_str == ":" && PP.getSpelling(Toks[i+1]) == ":") {
      ss << "::";
      i+=1;
    } else {
      ss << current_token_str << " ";
    }
  }
  ss << "\n";

  const auto optimizer_provided =
      ss.str().find("decompose_optimizer = ") != std::string::npos;
  if (optimizer_provided) {
    ss << "auto decomposed_program = "
          "__internal__::decompose_unitary(decompose_algo_name, "
       << var_name << ", decompose_buffer, decompose_optimizer);\n";
  } else {
    ss << "auto decomposed_program = "
          "__internal__::decompose_unitary(decompose_algo_name, "
       << var_name << ", decompose_buffer);\n";
  }
  // Add the qfast decomp and hook up to the qrt program.

  ss << "parent_kernel->addInstruction(decomposed_program);\n";
}

} // namespace qcor
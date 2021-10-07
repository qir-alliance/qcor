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
#include "quil_token_collector.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "clang/Basic/TokenKinds.h"

#include "xacc.hpp"
#include <iostream>
#include <memory>
#include <set>
#include <xacc_service.hpp>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL QuilTokenCollectorActivator : public BundleActivator {

public:
  QuilTokenCollectorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::QuilTokenCollector>();
    context.RegisterService<qcor::TokenCollector>(xt);
    // context.RegisterService<xacc::OptionsProvider>(acc);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QuilTokenCollectorActivator)

namespace qcor {

void QuilTokenCollector::collect(clang::Preprocessor &PP,
                                 clang::CachedTokens &Toks,
                                 std::stringstream &ss, const std::string &kernel_name) {
  bool inForLoop = false;
  for (int i = 0; i < Toks.size() - 1; i++) {

    auto token = PP.getSpelling(Toks[i]);

    ss << token << " ";

    auto nextToken = PP.getSpelling(Toks[i + 1]);

    // The following instructions in Quil
    // have different capitalization in XACC.
    if (nextToken == "RX")
      nextToken = "Rx";
    if (nextToken == "RY")
      nextToken = "Ry";
    if (nextToken == "RZ")
      nextToken = "Rz";
    if (nextToken == "CX")
      nextToken = "CNOT";

    // If the next token is an Instruction, then add newline
    if (xacc::hasService<xacc::Instruction>(nextToken)) {
      ss << "\n";
    }
  }
  ss << PP.getSpelling(Toks[Toks.size() - 1]) << "\n";
}

} // namespace qcor
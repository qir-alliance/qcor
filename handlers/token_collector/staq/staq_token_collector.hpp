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
#ifndef QCOR_HANDLERS_STAQTOKENCOLLECTOR_HPP_
#define QCOR_HANDLERS_STAQTOKENCOLLECTOR_HPP_

#include "token_collector.hpp"

namespace qcor {
class StaqTokenCollector : public TokenCollector {
public:
  void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
               std::vector<std::string> bufferNames,
               std::stringstream &ss, const std::string &kernel_name) override;
  const std::string name() const override { return "staq"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

#endif
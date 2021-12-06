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
#pragma once

#include "clang/Parse/Parser.h"

using namespace clang;

namespace qcor {


class Qasm3SyntaxHandler : public SyntaxHandler {
public:
  Qasm3SyntaxHandler() : SyntaxHandler("qasm3") {}
  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override;
  
 
  void AddToPredefines(llvm::raw_string_ostream &OS) override;
};
} // namespace qcor
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
#include <memory>

#include "llvm/IR/LLVMContext.h"

namespace clang {
class CodeGenAction;
}

namespace qcor {

std::unique_ptr<clang::CodeGenAction> emit_llvm_ir(
    const std::string src_code, std::vector<std::string> extra_headers = {});

}
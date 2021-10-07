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
#ifndef QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_
#define QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_

#include <sstream>

#include "clang/Parse/Parser.h"
#include "clang/Sema/DeclSpec.h"

namespace qcor {
void append_kernel(const std::string name,
                   const std::vector<std::string> &program_arg_types,
                   const std::vector<std::string> &program_parameters);
std::string run_token_collector(
    clang::Preprocessor &PP, clang::CachedTokens &Toks, std::string& src_to_prepend, const std::string kernel_name,
    const std::vector<std::string> &program_arg_types,
    const std::vector<std::string> &program_parameters,
    std::vector<std::string> bufferNames);

std::string run_token_collector(
    clang::Preprocessor &PP, clang::CachedTokens &Toks, 
    std::vector<std::string> bufferNames);

void set_verbose(bool verbose);
void info(const std::string &s);

}  // namespace qcor

#endif
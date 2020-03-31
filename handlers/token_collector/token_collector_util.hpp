#ifndef QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_
#define QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_

#include "clang/Parse/Parser.h"
#include <sstream>

namespace qcor {

// This goal of this function is to take the provided
// Tokens and figure out what source language the kernel is
// written in, and return the XACC quantum kernel + the name of the compiler.
std::pair<std::string, std::string>
run_token_collector(clang::Preprocessor &PP, clang::CachedTokens &Toks,
                    const std::string &function_prototype);

} // namespace qcor

#endif
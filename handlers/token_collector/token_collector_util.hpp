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

void map_xacc_kernel_to_qrt_calls(const std::string &kernel_str, const std::string& qpu_name,
                                  const std::string &compiler_name,
                                  const std::string &kernel_name,
                                  std::vector<std::string> bufferNames,
                                  llvm::raw_string_ostream &OS, int shots = 0);

void set_verbose(bool verbose);
void info(const std::string &s);

} // namespace qcor

#endif
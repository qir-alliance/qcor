#ifndef QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_
#define QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_

#include "clang/Parse/Parser.h"
#include "clang/Sema/DeclSpec.h"
#include <sstream>

namespace qcor {

std::string run_token_collector(clang::Preprocessor &PP,
                                clang::CachedTokens &Toks,
                                std::vector<std::string> bufferNames,
                                const std::string &function_prototype);

void run_token_collector_llvm_rt(clang::Preprocessor &PP,
                                 clang::CachedTokens &Toks,
                                 const std::string &function_prototype,
                                 std::vector<std::string> bufferNames,
                                 const std::string &kernel_name,
                                 llvm::raw_string_ostream &OS,
                                 const std::string &qpu_name, int shots = 0);

void set_verbose(bool verbose);
void info(const std::string &s);

} // namespace qcor

#endif
#ifndef QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_
#define QCOR_HANDLERS_TOKENCOLLECTORUTIL_HPP_

#include "clang/Parse/Parser.h"
#include "clang/Sema/DeclSpec.h"
#include <sstream>

namespace qcor {
void append_kernel(const std::string name);
std::string run_token_collector(clang::Preprocessor &PP,
                                clang::CachedTokens &Toks,
                                std::vector<std::string> bufferNames);

void set_verbose(bool verbose);
void info(const std::string &s);

} // namespace qcor

#endif
#include "token_collector_util.hpp"
#include "token_collector.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

#include <limits>
#include <qalloc>

#include "qrt_mapper.hpp"
#include "qrt.hpp"

#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Token.h"
#include "clang/Sema/DeclSpec.h"

namespace qcor {
void append_kernel(const std::string name) {
  ::quantum::kernels_in_translation_unit.push_back(name);
}

void set_verbose(bool verbose) { xacc::set_verbose(verbose); }
void info(const std::string &s) { xacc::info(s); }

std::string run_token_collector(clang::Preprocessor &PP,
                                clang::CachedTokens &Toks,
                                std::vector<std::string> bufferNames) {

  if (!xacc::isInitialized()) {
    xacc::Initialize();
  }

  // Loop through and partition Toks into language sets
  // for each language set, run the appropriate TokenCollector
  std::stringstream code_ss;

  std::string tc_name = "xasm";
  auto token_collector = xacc::getService<TokenCollector>("xasm");
  clang::CachedTokens tmp_cache;
  for (std::size_t i = 0; i < Toks.size(); i++) {
    // FIXME if using and using qcor::LANG;
    if (PP.getSpelling(Toks[i]) == "using") { //}.is(clang::tok::kw_using)) {
      // Flush the current CachedTokens...
      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss);

      i += 2;
      std::string lang_name = "";
      if (Toks[i].is(clang::tok::coloncolon)) {
        lang_name = PP.getSpelling(Toks[i + 1]);
        i += 3;
      } else {
        lang_name = PP.getSpelling(Toks[i + 2]);
        i += 4;
      }

      if (lang_name == "openqasm") {
        lang_name = "staq";
      }
      if (!xacc::hasService<TokenCollector>(lang_name)) {
        xacc::error("Invalid token collector name " + lang_name);
      }

      token_collector = xacc::getService<TokenCollector>(lang_name);
      tmp_cache.clear();
    }

    tmp_cache.push_back(Toks[i]);
  }

  if (!tmp_cache.empty()) {
    // Flush the current CachedTokens...
    token_collector->collect(PP, tmp_cache, bufferNames, code_ss);
  }

  return code_ss.str();
}

} // namespace qcor

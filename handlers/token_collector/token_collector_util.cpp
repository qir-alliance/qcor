#include "token_collector_util.hpp"
#include "token_collector.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

#include <limits>
#include <qalloc>

#include "qrt.hpp"
#include "qrt_mapper.hpp"

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
    } else if (PP.getSpelling(Toks[i]) == "decompose") {
      // Flush the current CachedTokens...
      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss);

      auto last_tc = token_collector;
      token_collector = xacc::getService<TokenCollector>("unitary");
      tmp_cache.clear();

      // skip decompose
      i++;

      // must open scope
      if (Toks[i].isNot(clang::tok::l_brace)) {
        xacc::error("Invalid decompose statement, must be of form decompose "
                    "{...} (...); with args in parenthesis being optional");
      }

      // skip l_brace
      i++; 

      // slurp up all tokens in the decompose scope {...}(...);
      int l_brace_count = 1;
      while (true) {
        if (Toks[i].is(clang::tok::l_brace)) {
          l_brace_count++;
        }
        if (Toks[i].is(clang::tok::r_brace)) {
          l_brace_count--;
        }
        if (l_brace_count == 0) {
          break;
        }
        tmp_cache.push_back(Toks[i]);
        i++;
      }

      // advance past the r_brace
      i++;
      if (Toks[i].isNot(clang::tok::l_paren)) {
        xacc::error("Invalid decompose statement, after scope close you must "
                    "provide at least buffer_name argument.");
      }

      // get the args, can be
      // (buffer_name)
      // (buffer_name, decompose_algo_name)
      // (buffer_name, decompose_algo_name, optimizer)
      i++;
      std::vector<std::string> arguments;
      while (Toks[i].isNot(clang::tok::r_paren)) {
        if (Toks[i].isNot(clang::tok::comma)) {
          arguments.push_back(PP.getSpelling(Toks[i]));
        }
        i++;
      }

      if (arguments.size() == 0) {
          xacc::error("Invalid decompose arguments, must at least provide the qreg variable");
      }
      
      if (arguments.size() == 1) {
          arguments.push_back("QFAST");
      }

      
      std::map<int, std::function<void(const std::string arg)>> arg_to_action{
          {0,
           [&](const std::string arg) {
             code_ss << "auto decompose_buffer_name = " << arg << ".name();\n";
           }
           },
          {1,
           [&](const std::string arg) {
             code_ss << "auto decompose_algo_name = \"" << arg << "\";\n";
           }
           },
          {2, [&](const std::string arg) {
             code_ss << "auto decompose_optimizer = " << arg << ";\n";
           }
           }
      };
      for (int i = 0; i < arguments.size(); i++) {
        arg_to_action[i](arguments[i]);
      }

      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss);

      // advance past the r_paren and semi colon
      i+=2;
      tmp_cache.clear();
      token_collector = last_tc;
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

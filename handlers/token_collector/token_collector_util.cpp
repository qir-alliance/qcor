#include "token_collector_util.hpp"

#include <limits>
#include <qalloc>

#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Token.h"
#include "clang/Sema/DeclSpec.h"
#include "qcor_config.hpp"
#include "qrt.hpp"
#include "qrt_mapper.hpp"
#include "token_collector.hpp"
#include "token_collector_helper.hpp"
#include "xacc.hpp"
#include "xacc_config.hpp"
#include "xacc_service.hpp"

namespace qcor {
void append_kernel(const std::string name,
                   const std::vector<std::string> &program_arg_types,
                   const std::vector<std::string> &program_parameters) {
  ::quantum::kernels_in_translation_unit.push_back(name);
  ::quantum::kernel_signatures_in_translation_unit[name] =
      std::make_pair(program_arg_types, program_parameters);
}

void set_verbose(bool verbose) { xacc::set_verbose(verbose); }
void info(const std::string &s) { xacc::info(s); }
std::string run_token_collector(clang::Preprocessor &PP,
                                clang::CachedTokens &Toks,
                                std::vector<std::string> bufferNames) {
  std::string s;
  std::vector<std::string> ss;
  return run_token_collector(PP, Toks, s, s, ss, ss, bufferNames);
}

std::string run_token_collector(
    clang::Preprocessor &PP, clang::CachedTokens &Toks,
    std::string &src_to_prepend, const std::string kernel_name,
    const std::vector<std::string> &program_arg_types,
    const std::vector<std::string> &program_parameters,
    std::vector<std::string> bufferNames) {
  if (!xacc::isInitialized()) {
    // Check if we are installed somewhere other than xacc install dir
    // if so add qcor plugins path to plugin search path
    if (std::string(XACC_ROOT) != std::string(QCOR_INSTALL_DIR)) {
      xacc::addPluginSearchPath(std::string(QCOR_INSTALL_DIR) +
                                std::string("/plugins"));
    }

    // if XACC_INSTALL_DIR != XACC_ROOT
    // then we need to pass --xacc-root-path XACC_ROOT to xacc Initialize
    //
    // Example - we are on Rigetti QCS and can't install via sudo
    // so we dpkg -x xacc to a user directory, but deb package
    // expects to be extracted to /usr/local/xacc, and xacc_config.hpp
    // points to that /usr/local/xacc. Therefore ServiceRegistry fails
    // to load plugins and libs, unless we change rootPath.
    std::string xacc_config_install_dir(XACC_INSTALL_DIR);
    std::string qcor_config_xacc_root(XACC_ROOT);
    if (xacc_config_install_dir != qcor_config_xacc_root) {
      std::vector<std::string> cmd_line{"--xacc-root-path",
                                        qcor_config_xacc_root};
      xacc::Initialize(cmd_line);
    } else {
      xacc::Initialize();
    }
  }

  // Loop through and partition Toks into language sets
  // for each language set, run the appropriate TokenCollector
  std::stringstream code_ss;

  std::string tc_name = "xasm";
  auto token_collector = xacc::getService<TokenCollector>("xasm");
  clang::CachedTokens tmp_cache;
  for (std::size_t i = 0; i < Toks.size(); i++) {
    // FIXME if using and using qcor::LANG;
    if (PP.getSpelling(Toks[i]) == "using") {  //}.is(clang::tok::kw_using)) {
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

    if (PP.getSpelling(Toks[i]) == "decompose") {
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
        xacc::error(
            "Invalid decompose statement, must be of form decompose "
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
        xacc::error(
            "Invalid decompose statement, after scope close you must "
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
        xacc::error(
            "Invalid decompose arguments, must at least provide the qreg "
            "variable");
      }

      if (arguments.size() == 1) {
        arguments.push_back("QFAST");
      }

      code_ss << "{\n";

      std::map<int, std::function<void(const std::string arg)>> arg_to_action{
          {0,
           [&](const std::string arg) {
             code_ss << "auto decompose_buffer_name = " << arg << ".name();\n";
           }},
          {1,
           [&](const std::string arg) {
             code_ss << "auto decompose_algo_name = \"" << arg << "\";\n";
           }},
          {2, [&](const std::string arg) {
             code_ss << "auto decompose_optimizer = " << arg << ";\n";
           }}};
      for (int i = 0; i < arguments.size(); i++) {
        arg_to_action[i](arguments[i]);
      }

      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss);

      code_ss << "}\n";

      // advance past the r_paren and semi colon
      i += 1;
      tmp_cache.clear();
      token_collector = last_tc;
      continue;
    }

    if (PP.getSpelling(Toks[i]) == "compute") {
      // Flush the current CachedTokens...
      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss);

      // auto last_tc = token_collector;
      // token_collector = xacc::getService<TokenCollector>("unitary");
      tmp_cache.clear();

      // skip compute
      i++;

      // must open scope
      if (Toks[i].isNot(clang::tok::l_brace)) {
        xacc::error(
            "Invalid decompose statement, must be of form decompose "
            "{...} (...); with args in parenthesis being optional");
      }

      // skip l_brace
      i++;

      // slurp up all tokens in the compute scope {...}
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

      // HANDLE THE TOKENS in tmp_cache
      // Take them, pass them to qcor syntax handler, to get back
      // new class source code (like we do in jit),
      std::stringstream dd;
      // FIXME WHAT IF MULTIPLE IN KERNEL
      std::string internal_kernel_function_name =
          "__internal__compute_context_" + kernel_name;
      
        std::stringstream tmpss;
       if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, tmpss);
 
      auto src_code = __internal__::qcor::construct_kernel_subtype(
          tmpss.str(), internal_kernel_function_name, program_arg_types,
          program_parameters, bufferNames);
      src_to_prepend = src_code;

      code_ss << "::quantum::qrt_impl->__begin_mark_segment_as_compute();\n";
      code_ss << internal_kernel_function_name << "(parent_kernel, " << program_parameters[0];
      for (int i = 1; i < program_parameters.size(); i++) {
        code_ss << ", " << program_parameters[i];
      }
      code_ss << ");\n";
      code_ss << "::quantum::qrt_impl->__end_mark_segment_as_compute();\n";

      if (PP.getSpelling(Toks[i]) != "action") {
        xacc::error(
            "Invalid compute-action statement, after scope close you must "
            "provide the action scope {...}.");
      }

      i += 2;

      tmp_cache.clear();

      // slurp up all tokens in the action {...};
      l_brace_count = 1;
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

      // HANDLE THE TOKENS IN ACTION
      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss);

      code_ss << "::quantum::qrt_impl->__begin_mark_segment_as_compute();\n";
      code_ss << internal_kernel_function_name << "::adjoint(parent_kernel, "
              << program_parameters[0];
      for (int i = 1; i < program_parameters.size(); i++) {
        code_ss << ", " << program_parameters[i];
      }
      code_ss << ");\n";
      code_ss << "::quantum::qrt_impl->__end_mark_segment_as_compute();\n";
      tmp_cache.clear();
      continue;
    }

    tmp_cache.push_back(Toks[i]);
  }

  if (!tmp_cache.empty()) {
    // Flush the current CachedTokens...
    token_collector->collect(PP, tmp_cache, bufferNames, code_ss);
  }

  return code_ss.str();
}

}  // namespace qcor

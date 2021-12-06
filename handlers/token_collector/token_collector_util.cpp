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
#include "token_collector_util.hpp"

#include <limits>
#include <qalloc>
#include <regex>

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
  // Just ignored if we have tracked this kernel already.
  if (!xacc::container::contains(::quantum::kernels_in_translation_unit,
                                 name)) {
    ::quantum::kernels_in_translation_unit.push_back(name);
    ::quantum::kernel_signatures_in_translation_unit[name] =
        std::make_pair(program_arg_types, program_parameters);
  }
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
  int compute_counter = 0;

  std::string tc_name = "xasm";
  auto token_collector = xacc::getService<TokenCollector>("xasm");
  clang::CachedTokens tmp_cache;
  for (std::size_t i = 0; i < Toks.size(); i++) {
    // FIXME if using and using qcor::LANG;
    if (PP.getSpelling(Toks[i]) == "using") {  //}.is(clang::tok::kw_using)) {
      // Flush the current CachedTokens...
      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss,
                                 kernel_name);

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
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss,
                                 kernel_name);

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
             // Cache the decompose buffer as well
             code_ss << "auto decompose_buffer = " << arg << ";\n";
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
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss,
                                 kernel_name);

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
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss,
                                 kernel_name);

      // auto last_tc = token_collector;
      // token_collector = xacc::getService<TokenCollector>("unitary");
      tmp_cache.clear();

      // skip compute
      i++;

      // must open scope
      if (Toks[i].isNot(clang::tok::l_brace)) {
        xacc::error(
            "Invalid compute statement, must be of form decompose "
            "{...} (...); with args in parenthesis being optional: " +
            PP.getSpelling(Toks[i]));
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
      auto counter_str = std::to_string(compute_counter);
      std::stringstream dd;
      std::string internal_kernel_function_name =
          "__internal__compute_context_" + kernel_name + "_" + counter_str;

      std::stringstream tmpss;
      if (!tmp_cache.empty())
        token_collector->collect(PP, tmp_cache, bufferNames, tmpss,
                                 internal_kernel_function_name);

      // Need to handle compute-action capture variables...
      // Strategy: add function<void(Args...)> type to beginning of arg_types.
      // After kernel code is generated, replace tmpss.str() with
      // std::get<0>(args_tuple)(args...).

      std::string functor_type =
          "std::function<void(std::shared_ptr<CompositeInstruction>, " +
          program_arg_types[0];
      for (int i = 1; i < program_arg_types.size(); i++)
        functor_type += ", " + program_arg_types[i];
      functor_type += ")>";
      std::vector<std::string> mutable_arg_types = program_arg_types,
                               mutable_parameters = program_parameters;
      mutable_arg_types.insert(mutable_arg_types.begin(), functor_type);
      mutable_parameters.insert(
          mutable_parameters.begin(),
          "__" + kernel_name + "_" + counter_str + "__compute_functor");

      auto src_code = ::__internal__::qcor::construct_kernel_subtype(
          tmpss.str(), internal_kernel_function_name, mutable_arg_types,
          mutable_parameters, bufferNames);

      std::string replace_csp_code =
          "std::get<0>(args_tuple)(parent_kernel," + program_parameters[0];
      for (int i = 1; i < program_parameters.size(); i++)
        replace_csp_code += ", " + program_parameters[i];
      replace_csp_code += ");\n";

      // Replace the operator()(...) contents with a call
      // to the provided functor
      auto find_index = src_code.find(tmpss.str());
      src_code.replace(find_index, tmpss.str().length(), replace_csp_code);
      src_to_prepend += src_code;

      // Construct the compute context functor
      std::string functor_code =
          "const auto __" + kernel_name + "_" + counter_str +
          "__compute_functor = [&](std::shared_ptr<CompositeInstruction> "
          "parent_kernel, " +
          program_arg_types[0] + " " + program_parameters[0];
      for (int i = 1; i < program_parameters.size(); i++)
        functor_code +=
            ", " + program_arg_types[i] + " " + program_parameters[i];
      functor_code += ") {\n" + tmpss.str() + "\n};\n";

      code_ss << functor_code;
      code_ss << "::quantum::qrt_impl->__begin_mark_segment_as_compute();\n";
      code_ss << internal_kernel_function_name << "(parent_kernel, "
              << "__" + kernel_name + "_" + counter_str + "__compute_functor, "
              << program_parameters[0];

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
        token_collector->collect(PP, tmp_cache, bufferNames, code_ss,
                                 kernel_name);

      code_ss << "::quantum::qrt_impl->__begin_mark_segment_as_compute();\n";
      code_ss << internal_kernel_function_name << "::adjoint(parent_kernel, "
              << "__" + kernel_name + "_" + counter_str + "__compute_functor, "
              << program_parameters[0];
      for (int i = 1; i < program_parameters.size(); i++) {
        code_ss << ", " << program_parameters[i];
      }
      code_ss << ");\n";
      code_ss << "::quantum::qrt_impl->__end_mark_segment_as_compute();\n";
      tmp_cache.clear();
      compute_counter++;
      continue;
    }

    tmp_cache.push_back(Toks[i]);
  }

  if (!tmp_cache.empty()) {
    // Flush the current CachedTokens...
    token_collector->collect(PP, tmp_cache, bufferNames, code_ss, kernel_name);
  }

  return code_ss.str();
}

}  // namespace qcor

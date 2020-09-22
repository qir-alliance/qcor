#pragma once
#include "IRProvider.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xasm_singleVisitor.h"
#include <regex>

using namespace xasm;

std::map<std::string, std::string> common_name_map{
    {"CX", "CNOT"}, {"qcor::exp", "exp_i_theta"}, {"exp", "exp_i_theta"}};
using xasm_single_result_type =
    std::pair<std::string, std::shared_ptr<xacc::Instruction>>;

class xasm_single_visitor : public xasm::xasm_singleVisitor {
protected:
  int n_cached_execs = 0;

public:
  xasm_single_result_type result;

  antlrcpp::Any
  visitStatement(xasm_singleParser::StatementContext *context) override {
    // should only have 1 child, if it is qinst
    // we expect a xacc Instruction return type
    // if cinst we expect a Cinst
    return visitChildren(context);
  }

  antlrcpp::Any visitQinst(xasm_singleParser::QinstContext *context) override {
    if (!xacc::isInitialized()) {
      xacc::Initialize();
    }

    // if not in instruction registry, then forward to classical instructions
    auto inst_name = context->inst_name->getText();
    auto provider = xacc::getIRProvider("quantum");

    if (common_name_map.count(inst_name)) {
      inst_name = common_name_map[inst_name];
    }

    if (xacc::container::contains(provider->getInstructions(), inst_name)) {

      // We don't really care about Instruction::bits(), qrt_mapper
      // will look for bit expressions and use those, so just set
      // everything as a string...

      // Create an instance of the Instruction with the given name
      auto inst = provider->createInstruction(inst_name, 0);

      // If it is not composite, look for its bit expressions
      // and parameter expressions
      if (!inst->isComposite()) {
        // Get the number of required bits and parameters
        auto required_bits = inst->nRequiredBits();
        auto required_params = inst->getParameters().size();

        if (required_bits + required_params !=
                context->explist()->exp().size() &&
            inst_name != "Measure") {
          std::stringstream xx;
          xx << "Invalid quantum instruction expression. " << inst_name
             << " requires " << required_bits << " qubit args and "
             << required_params << " parameter args.";
          xacc::error(xx.str());
        }

        // Get the qubit expresssions
        std::vector<std::string> buffer_names;
        for (int i = 0; i < required_bits; i++) {
          auto bit_expr = context->explist()->exp(i);
          auto bit_expr_str = bit_expr->getText();

          auto found_bracket = bit_expr_str.find_first_of("[");
          if (found_bracket != std::string::npos) {
            auto buffer_name = bit_expr_str.substr(0, found_bracket);
            auto bit_idx_expr = bit_expr_str.substr(
                found_bracket + 1, bit_expr_str.length() - found_bracket - 2);

            buffer_names.push_back(buffer_name);
            inst->setBitExpression(i, bit_idx_expr);
          } else {
            xacc::error("Must provide qreg[IDX] and not just qreg.");
          }
        }
        inst->setBufferNames(buffer_names);

        // Get the parameter expressions
        int counter = 0;
        for (int i = required_bits; i < context->explist()->exp().size(); i++) {
          inst->setParameter(counter, context->explist()->exp(i)->getText());
          counter++;
        }
      } else {

        // I don't want to use xasm circuit gen any more...
        // So use it as a fallback, but first look for previous
        if (xacc::container::contains(quantum::kernels_in_translation_unit,
                                      context->inst_name->getText())) {
          // If this is a previously seen quantum kernel
          // then we want to update its signature to add the
          // parent CompositeInstruction argument
          std::stringstream ss;
          for (auto c : context->children) {
            if (c->getText() == "(") {
              ss << c->getText() << "parent_kernel, ";

            } else {
              ss << c->getText() << " ";
            }
          }

          result.first = ss.str() + "\n";
          return 0;
        } else {
          // this is something like exp_i_theta(q,...);
          auto comp_inst = xacc::ir::asComposite(inst);
          inst->setBufferNames({context->explist()->exp(0)->getText()});
          for (int i = 1; i < context->explist()->exp().size(); i++) {
            comp_inst->addArgument(context->explist()->exp(i)->getText(), "");
          }
        }
      }

      result.second = inst;
    } else {

      std::stringstream ss;

      if (xacc::container::contains(quantum::kernels_in_translation_unit,
                                    context->inst_name->getText())) {
        // If this is a previously seen quantum kernel
        // then we want to update its signature to add the
        // parent CompositeInstruction argument

        for (auto c : context->children) {
          if (c->getText() == "(") {
            ss << c->getText() << "parent_kernel, ";

          } else {
            ss << c->getText() << " ";
          }
        }
      } else {
        for (auto c : context->children) {
          ss << c->getText() << " ";
        }
      }
      result.first = ss.str() + "\n";
      n_cached_execs++;
    }

    return 0;
  }

  antlrcpp::Any visitCinst(xasm_singleParser::CinstContext *context) override {

    // Strategy here is simple, we just want to
    // preserve all classical code statements in
    // the original quantum kernel

    std::stringstream ss;

    if (context->getText().find("::adjoint") != std::string::npos) {
      for (auto c : context->children) {
        if (c->getText() == "(") {
          ss << c->getText() << "parent_kernel, ";

        } else {
          ss << c->getText() << " ";
        }
      }
    } else if (context->getText().find("::ctrl") != std::string::npos) {
      for (auto c : context->children) {
        if (c->getText() == "(") {
          ss << c->getText() << "parent_kernel, ";

        } else {
          ss << c->getText() << " ";
        }
      }
    } else if (context->getText().find("Measure") != std::string::npos) {
      // Found measure in a classical instruction.
      // std::cout << "FOUND MEAS: " << context->getText() << "\n";
      // To be extra careful, we use search and replace to handle edge case
      // whereby `!Measure` is considered as 1 token.
      const auto replaceAll = [](std::string &s, const std::string &search,
                                 const std::string &replace) {
        for (size_t pos = 0;; pos += replace.length()) {
          pos = s.find(search, pos);
          if (pos == std::string::npos) {
            break;
          }
          if ((s.size() > pos + search.size()) &&
              // If "Measure" is not followed by a space or '(',
              // i.e. not having a function call signature,
              // we don't replace.
              (!isspace(s[pos + search.length()]) ||
               (s[pos + search.length()] != '('))) {
            continue;
          }
          s.erase(pos, search.length());
          s.insert(pos, replace);
        }
      };
      for (auto c : context->children) {
        auto origText = c->getText();
        replaceAll(origText, "Measure", " quantum::mz");
        ss << origText << " ";
      }
    } else {
      for (auto c : context->children) {
        ss << c->getText() << " ";
      }
    }

    result.first = ss.str() + "\n";
    return 0;
  }

  antlrcpp::Any visitLine(xasm_singleParser::LineContext *context) override {
    return 0;
  }

  antlrcpp::Any
  visitComment(xasm_singleParser::CommentContext *context) override {
    return 0;
  }
  antlrcpp::Any
  visitCompare(xasm_singleParser::CompareContext *context) override {
    return 0;
  }

  antlrcpp::Any
  visitCpp_type(xasm_singleParser::Cpp_typeContext *context) override {
    return 0;
  }

  antlrcpp::Any
  visitExplist(xasm_singleParser::ExplistContext *context) override {
    return 0;
  }

  antlrcpp::Any visitExp(xasm_singleParser::ExpContext *context) override {
    return 0;
  }

  antlrcpp::Any
  visitUnaryop(xasm_singleParser::UnaryopContext *context) override {
    return 0;
  }

  antlrcpp::Any visitId(xasm_singleParser::IdContext *context) override {
    return 0;
  }

  antlrcpp::Any visitReal(xasm_singleParser::RealContext *context) override {
    return 0;
  }

  antlrcpp::Any
  visitString(xasm_singleParser::StringContext *context) override {
    return 0;
  }
};
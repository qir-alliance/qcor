#pragma once
#include <regex>

#include "IRProvider.hpp"
#include "pyxasmBaseVisitor.h"
#include "qrt.hpp"
#include "xacc.hpp"

using namespace pyxasm;

std::map<std::string, std::string> common_name_map{
    {"CX", "CNOT"}, {"qcor::exp", "exp_i_theta"}, {"exp", "exp_i_theta"}};

using pyxasm_result_type =
    std::pair<std::string, std::shared_ptr<xacc::Instruction>>;

class pyxasm_visitor : public pyxasmBaseVisitor {
 protected:
  std::shared_ptr<xacc::IRProvider> provider;
  // List of buffers in the *context* of this XASM visitor
  std::vector<std::string> bufferNames;

 public:
  pyxasm_visitor(const std::vector<std::string> &buffers = {})
      : provider(xacc::getIRProvider("quantum")), bufferNames(buffers) {}
  pyxasm_result_type result;

  bool in_for_loop = false;

  antlrcpp::Any visitAtom_expr(
      pyxasmParser::Atom_exprContext *context) override {
      
    // Handle kernel::ctrl(...), kernel::adjoint(...)
    if (!context->trailer().empty() &&
        (context->trailer()[0]->getText() == ".ctrl" ||
         context->trailer()[0]->getText() == ".adjoint")) {
      std::cout << "HELLO: " << context->getText() << "\n";
      std::cout << context->trailer()[0]->getText() << "\n";
      std::cout << context->atom()->getText() << "\n";

      std::cout << context->trailer()[1]->getText() << "\n";
      std::cout << context->trailer()[1]->arglist() << "\n";
      auto arg_list = context->trailer()[1]->arglist();

      std::stringstream ss;
      // Remove the first '.' character
      const std::string methodName = context->trailer()[0]->getText().substr(1);
      ss << context->atom()->getText() << "::" << methodName
         << "(parent_kernel";
      for (int i = 0; i < arg_list->argument().size(); i++) {
        ss << ", " << arg_list->argument(i)->getText();
      }
      ss << ");\n";

      std::cout << "HELLO SS: " << ss.str() << "\n";
      result.first = ss.str();
      return 0;
    }
    if (context->atom()->NAME() != nullptr) {
      auto inst_name = context->atom()->NAME()->getText();

      if (common_name_map.count(inst_name)) {
        inst_name = common_name_map[inst_name];
      }

      if (xacc::container::contains(provider->getInstructions(), inst_name)) {
        // Create an instance of the Instruction with the given name
        auto inst = provider->createInstruction(inst_name, 0);

        // If it is not composite, look for its bit expressions
        // and parameter expressions
        if (!inst->isComposite()) {
          // Get the number of required bits and parameters
          auto required_bits = inst->nRequiredBits();
          auto required_params = inst->getParameters().size();

          if (!context->trailer().empty()) {
            auto atom_n_args =
                context->trailer()[0]->arglist()->argument().size();

            if (required_bits + required_params != atom_n_args &&
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
              auto bit_expr = context->trailer()[0]->arglist()->argument()[i];
              auto bit_expr_str = bit_expr->getText();

              auto found_bracket = bit_expr_str.find_first_of("[");
              if (found_bracket != std::string::npos) {
                auto buffer_name = bit_expr_str.substr(0, found_bracket);
                auto bit_idx_expr = bit_expr_str.substr(
                    found_bracket + 1,
                    bit_expr_str.length() - found_bracket - 2);
                buffer_names.push_back(buffer_name);
                inst->setBitExpression(i, bit_idx_expr);
              } else {
                xacc::error("Must provide qreg[IDX] and not just qreg.");
              }
            }
            inst->setBufferNames(buffer_names);

            // Get the parameter expressions
            int counter = 0;
            for (int i = required_bits; i < atom_n_args; i++) {
              inst->setParameter(counter,
                                 replacePythonConstants(context->trailer()[0]
                                                            ->arglist()
                                                            ->argument()[i]
                                                            ->getText()));
              counter++;
            }
          }
          result.second = inst;
        } else {
          // Composite instructions, e.g. exp_i_theta
          if (inst_name == "exp_i_theta") {
            // Expected 3 params:
            if (context->trailer()[0]->arglist()->argument().size() != 3) {
              xacc::error(
                  "Invalid number of arguments for the 'exp_i_theta' "
                  "instruction. Expected 3, got " +
                  std::to_string(
                      context->trailer()[0]->arglist()->argument().size()) +
                  ". Please check your input.");
            }

            std::stringstream ss;
            // Delegate to the QRT call directly.
            ss << "quantum::exp("
               << context->trailer()[0]->arglist()->argument(0)->getText()
               << ", "
               << context->trailer()[0]->arglist()->argument(1)->getText()
               << ", "
               << context->trailer()[0]->arglist()->argument(2)->getText()
               << ");\n";
            result.first = ss.str();
          }
          // Handle potential name collision: user-defined kernel having the
          // same name as an XACC circuit: e.g. common names such as qft, iqft
          // Note: these circuits (except exp_i_theta) don't have QRT
          // equivalents.
          // Condition: first argument is a qubit register
          else if (!context->trailer()[0]->arglist()->argument().empty() &&
                   xacc::container::contains(bufferNames, context->trailer()[0]
                                                              ->arglist()
                                                              ->argument(0)
                                                              ->getText())) {
            std::stringstream ss;
            // Use the kernel call with a parent kernel arg.
            ss << inst_name << "(parent_kernel, ";
            const auto &argList = context->trailer()[0]->arglist()->argument();
            for (size_t i = 0; i < argList.size(); ++i) {
              ss << argList[i]->getText();
              if (i != argList.size() - 1) {
                ss << ", ";
              }
            }
            ss << ");\n";
            result.first = ss.str();
          } else {
            xacc::error("Composite instruction '" + inst_name +
                        "' is not currently supported.");
          }
        }
      } else {
        // This kernel *callable* is not an intrinsic instruction, just
        // reassemble the call:
        // Check that the *first* argument is a *qreg* in the current context of
        // *this* kernel.
        if (!context->trailer().empty() &&
            !context->trailer()[0]->arglist()->argument().empty() &&
            xacc::container::contains(
                bufferNames,
                context->trailer()[0]->arglist()->argument(0)->getText())) {
          std::stringstream ss;
          // Use the kernel call with a parent kernel arg.
          ss << inst_name << "(parent_kernel, ";
          // TODO: We potentially need to handle *inline* expressions in the
          // function call.
          const auto &argList = context->trailer()[0]->arglist()->argument();
          for (size_t i = 0; i < argList.size(); ++i) {
            ss << argList[i]->getText();
            if (i != argList.size() - 1) {
              ss << ", ";
            }
          }
          ss << ");\n";
          result.first = ss.str();
        }
      }
    }
    return 0;
  }

  antlrcpp::Any visitFor_stmt(pyxasmParser::For_stmtContext *context) override {
    auto counter_expr = context->exprlist()->expr()[0];
    auto iter_container = context->testlist()->test()[0]->getText();
    // Rewrite:
    // Python: "for <var> in <expr>:"
    // C++: for (auto& var: <expr>) {}
    // Note: we add range(int) as a C++ function to support this common pattern.
    std::stringstream ss;
    ss << "for (auto &" << counter_expr->getText() << " : " << iter_container
       << ") {\n";
    result.first = ss.str();
    in_for_loop = true;
    return 0;
  }

  antlrcpp::Any visitExpr_stmt(pyxasmParser::Expr_stmtContext *ctx) override {
    if (ctx->ASSIGN().size() == 1 && ctx->testlist_star_expr().size() == 2) {
      // Handle simple assignment: a = expr
      std::stringstream ss;
      const std::string lhs = ctx->testlist_star_expr(0)->getText();
      const std::string rhs = replacePythonConstants(
          replaceMeasureAssignment(ctx->testlist_star_expr(1)->getText()));
      ss << "auto " << lhs << " = " << rhs << "; \n";
      result.first = ss.str();
      if (rhs.find("**") != std::string::npos) {
        // keep processing
        return visitChildren(ctx);
      } else {
        return 0;
      }
    } else {
      return visitChildren(ctx);
    }
  }

  antlrcpp::Any visitPower(pyxasmParser::PowerContext *context) override {
    if (context->getText().find("**") != std::string::npos &&
        context->factor() != nullptr) {
      // Here we handle x**y from parent assignment expression
      auto replaceAll = [](std::string &s, const std::string &search,
                           const std::string &replace) {
        for (std::size_t pos = 0;; pos += replace.length()) {
          // Locate the substring to replace
          pos = s.find(search, pos);
          if (pos == std::string::npos) break;
          // Replace by erasing and inserting
          s.erase(pos, search.length());
          s.insert(pos, replace);
        }
      };
      auto factor = context->factor();
      auto atom_expr = context->atom_expr();
      std::string s =
          "std::pow(" + atom_expr->getText() + ", " + factor->getText() + ")";
      replaceAll(result.first, context->getText(), s);
      return 0;
    }
    return visitChildren(context);
  }

  virtual antlrcpp::Any
  visitIf_stmt(pyxasmParser::If_stmtContext *ctx) override {
    // Only support single clause atm
    if (ctx->test().size() == 1) {
      std::stringstream ss;
      ss << "if ("
         << replacePythonConstants(
                replaceMeasureAssignment(ctx->test(0)->getText()))
         << ") {\n";
      result.first = ss.str();
      return 0;
    }
    return visitChildren(ctx);
  }

 private:
  // Replaces common Python constants, e.g. 'math.pi' or 'numpy.pi'.
  // Note: the library names have been resolved to their original names.
  std::string replacePythonConstants(const std::string &in_pyExpr) const {
    // List of all keywords to be replaced
    const std::map<std::string, std::string> REPLACE_MAP{{"math.pi", "M_PI"},
                                                         {"numpy.pi", "M_PI"}};
    std::string newSrc = in_pyExpr;
    for (const auto &[key, value] : REPLACE_MAP) {
      const auto pos = newSrc.find(key);
      if (pos != std::string::npos) {
        newSrc.replace(pos, key.length(), value);
      }
    }
    return newSrc;
  }

  // Assignment of Measure results -> variable or in if conditional statements
  std::string replaceMeasureAssignment(const std::string &in_expr) const {
    if (in_expr.find("Measure") != std::string::npos) {
      // Found measure in an if statement instruction.
      const auto replaceMeasureInst = [](std::string &s,
                                         const std::string &search,
                                         const std::string &replace) {
        for (size_t pos = 0;; pos += replace.length()) {
          pos = s.find(search, pos);
          if (pos == std::string::npos) {
            break;
          }
          if (!isspace(s[pos + search.length()]) &&
              (s[pos + search.length()] != '(')) {
            continue;
          }
          s.erase(pos, search.length());
          s.insert(pos, replace);
        }
      };

      std::string result = in_expr;
      replaceMeasureInst(result, "Measure", "quantum::mz");
      return result;
    } else {
      return in_expr;
    }
  }
};
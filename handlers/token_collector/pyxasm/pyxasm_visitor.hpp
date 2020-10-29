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

public:
  pyxasm_visitor() : provider(xacc::getIRProvider("quantum")) {}
  pyxasm_result_type result;

  bool in_for_loop = false;

  antlrcpp::Any
  visitAtom_expr(pyxasmParser::Atom_exprContext *context) override {
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
                auto bit_idx_expr = bit_expr_str.substr(found_bracket + 1,
                                                        bit_expr_str.length() -
                                                            found_bracket - 2);
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
          } else {
            xacc::error("Composite instruction '" + inst_name +
                        "' is not currently supported.");
          }
        }
      } else {
        // This *callable* is not an intrinsic instruction, just reassemble the
        // call:
        // TODO: validate that this is a *previously-defined* kernel (i.e. not a
        // classical function call)
        if (!context->trailer().empty()) {
          std::stringstream ss;
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

    if (context->testlist()->test()[0]->getText().find("range") !=
        std::string::npos) {
      auto range_str = context->testlist()->test()[0]->getText();
      auto found_paren = range_str.find_first_of("(");
      auto range_contents = range_str.substr(
          found_paren + 1, range_str.length() - found_paren - 2);

      std::stringstream ss;
      ss << "for (int " << counter_expr->getText() << " = 0; "
         << counter_expr->getText() << " < " << range_contents << "; ++"
         << counter_expr->getText() << " ) {\n";

      result.first = ss.str();
      in_for_loop = true;

    } else {
      xacc::error(
          "QCOR PyXasm can only handle 'for VAR in range(QREG.size())' at the "
          "moment.");
    }

    return 0;
  }

  antlrcpp::Any visitExpr_stmt(pyxasmParser::Expr_stmtContext *ctx) override {
    if (ctx->ASSIGN().size() == 1 && ctx->testlist_star_expr().size() == 2) {
      // Handle simple assignment: a = expr
      std::stringstream ss;
      const std::string lhs = ctx->testlist_star_expr(0)->getText();
      const std::string rhs = ctx->testlist_star_expr(1)->getText();
      ss << "auto " << lhs << " = " << rhs << "; \n";
      result.first = ss.str();
      return 0;
    } else {
      return visitChildren(ctx);
    }
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
};
#pragma once
#include <regex>

#include "IRProvider.hpp"
#include "pyxasmBaseVisitor.h"
#include "qcor_utils.hpp"
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
  // List of *declared* variables
  std::vector<std::string> declared_var_names;

 public:
  pyxasm_visitor(const std::vector<std::string> &buffers = {},
                 const std::vector<std::string> &local_var_names = {})
      : provider(xacc::getIRProvider("quantum")),
        bufferNames(buffers),
        declared_var_names(local_var_names) {}
  pyxasm_result_type result;
  // New var declared (auto type) after visiting this node.
  std::string new_var;
  bool in_for_loop = false;
  // Var to keep track of sub-node rewrite:
  // e.g., traverse down the AST recursively.
  std::stringstream sub_node_translation;
  bool is_processing_sub_expr = false;

  antlrcpp::Any visitAtom_expr(
      pyxasmParser::Atom_exprContext *context) override {
    // std::cout << "Atom_exprContext: " << context->getText() << "\n";
    // Strategy:
    // At the top level, we analyze the trailer to determine the 
    // list of function call arguments.
    // Then, traverse down the arg. node to see if there is a potential rewrite rules
    // e.g. for arrays (as testlist_comp nodes)
    // Otherwise, just get the argument text as is.
    /*
    atom_expr: (AWAIT)? atom trailer*;
    atom: ('(' (yield_expr|testlist_comp)? ')' |
       '[' (testlist_comp)? ']' |
       '{' (dictorsetmaker)? '}' |
       NAME | NUMBER | STRING+ | '...' | 'None' | 'True' | 'False');
    */
    // Only processes these for sub-expressesions, 
    // e.g. re-entries to this function
    if (is_processing_sub_expr) {
      if (context->atom() && context->atom()->OPEN_BRACK() &&
          context->atom()->CLOSE_BRACK() && context->atom()->testlist_comp()) {
        // Array type expression:
        // std::cout << "Array atom expression: "
        //           << context->atom()->testlist_comp()->getText() << "\n";
        // Use braces
        sub_node_translation << "{";
        bool firstElProcessed = false;
        for (auto &testNode : context->atom()->testlist_comp()->test()) {
          // std::cout << "Array elem: " << testNode->getText() << "\n";
          // Add comma if needed (there is a previous element)
          if (firstElProcessed) {
            sub_node_translation << ", ";
          }
          sub_node_translation << testNode->getText();
          firstElProcessed = true;
        }
        sub_node_translation << "}";
        return 0;
      }

      // We don't have a re-write rule for this one (py::dict)
      if (context->atom() && context->atom()->OPEN_BRACE() &&
          context->atom()->CLOSE_BRACE() && context->atom()->dictorsetmaker()) {
        // Dict:
        // std::cout << "Dict atom expression: "
        //           << context->atom()->dictorsetmaker()->getText() << "\n";
        // TODO:
        return 0;
      }

      if (context->atom() && !context->atom()->STRING().empty()) {
        // Strings:
        for (auto &strNode : context->atom()->STRING()) {
          std::string cppStrLiteral = strNode->getText();
          // Handle Python single-quotes
          if (cppStrLiteral.front() == '\'' && cppStrLiteral.back() == '\'') {
            cppStrLiteral.front() = '"';
            cppStrLiteral.back() = '"';
          }
          sub_node_translation << cppStrLiteral;
          // std::cout << "String expression: " << strNode->getText() << " --> "
          //           << cppStrLiteral << "\n";
        }
        return 0;
      }

      const auto isSliceOp =
          [](pyxasmParser::Atom_exprContext *atom_expr_context) -> bool {
        if (atom_expr_context->trailer().size() == 1) {
          auto subscriptlist = atom_expr_context->trailer(0)->subscriptlist();
          if (subscriptlist && subscriptlist->subscript().size() == 1) {
            auto subscript = subscriptlist->subscript(0);
            const auto nbTestTerms = subscript->test().size();
            // Multiple test terms (separated by ':')
            return (nbTestTerms > 1);
          }
        }

        return false;
      };

      // Handle slicing operations (multiple array subscriptions separated by
      // ':') on a qreg.
      if (context->atom() &&
          xacc::container::contains(bufferNames, context->atom()->getText()) &&
          isSliceOp(context)) {
        // std::cout << "Slice op: " << context->getText() << "\n";
        sub_node_translation << context->atom()->getText()
                             << ".extract_range({";
        auto subscripts =
            context->trailer(0)->subscriptlist()->subscript(0)->test();
        assert(subscripts.size() > 1);
        std::vector<std::string> subscriptTerms;
        for (auto &test : subscripts) {
          subscriptTerms.emplace_back(test->getText());
        }

        auto sliceOp =
            context->trailer(0)->subscriptlist()->subscript(0)->sliceop();
        if (sliceOp && sliceOp->test()) {
          subscriptTerms.emplace_back(sliceOp->test()->getText());
        }
        assert(subscriptTerms.size() == 2 || subscriptTerms.size() == 3);

        for (int i = 0; i < subscriptTerms.size(); ++i) {
          // Need to cast to prevent compiler errors,
          // e.g. when using q.size() which returns an int.
          sub_node_translation << "static_cast<size_t>(" << subscriptTerms[i]
                               << ")";
          if (i != subscriptTerms.size() - 1) {
            sub_node_translation << ", ";
          }
        }

        sub_node_translation << "})";

        // convert the slice op to initializer list:
        // std::cout << "Slice Convert: " << context->getText() << " --> "
        //           << sub_node_translation.str() << "\n";
        return 0;
      }

      return 0;
    }

    // Handle kernel::ctrl(...), kernel::adjoint(...)
    if (!context->trailer().empty() &&
        (context->trailer()[0]->getText() == ".ctrl" ||
         context->trailer()[0]->getText() == ".adjoint")) {
      // std::cout << "HELLO: " << context->getText() << "\n";
      // std::cout << context->trailer()[0]->getText() << "\n";
      // std::cout << context->atom()->getText() << "\n";

      // std::cout << context->trailer()[1]->getText() << "\n";
      // std::cout << context->trailer()[1]->arglist() << "\n";
      auto arg_list = context->trailer()[1]->arglist();

      std::stringstream ss;
      // Remove the first '.' character
      const std::string methodName = context->trailer()[0]->getText().substr(1);
      // If this is a *variable*, then using '.' for control/adjoint.
      // Otherwise, use '::' (global scope kernel names)
      const std::string separator =
          (xacc::container::contains(declared_var_names,
                                     context->atom()->getText()))
              ? "."
              : "::";

      ss << context->atom()->getText() << separator << methodName
         << "(parent_kernel";
      for (int i = 0; i < arg_list->argument().size(); i++) {
        ss << ", " << rewriteFunctionArgument(*(arg_list->argument(i)));
      }
      ss << ");\n";

      // std::cout << "HELLO SS: " << ss.str() << "\n";
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
              auto bit_expr_str = rewriteFunctionArgument(*bit_expr);

              auto found_bracket = bit_expr_str.find_first_of("[");
              if (found_bracket != std::string::npos) {
                auto buffer_name = bit_expr_str.substr(0, found_bracket);
                auto bit_idx_expr = bit_expr_str.substr(
                    found_bracket + 1,
                    bit_expr_str.length() - found_bracket - 2);
                buffer_names.push_back(buffer_name);
                inst->setBitExpression(i, bit_idx_expr);
              } else {
                // Indicate this is a qubit(-1) or a qreg(-2)
                inst->setBitExpression(-1, bit_expr_str);
                buffer_names.push_back(bit_expr_str);
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
          else if (xacc::container::contains(
                       ::quantum::kernels_in_translation_unit, inst_name) ||
                   !context->trailer()[0]->arglist()->argument().empty() &&
                       xacc::container::contains(bufferNames,
                                                 context->trailer()[0]
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
        // *this* kernel or the function name is a kernel in translation unit.
        if (xacc::container::contains(::quantum::kernels_in_translation_unit,
                                      inst_name) ||
            (!context->trailer().empty() && context->trailer()[0]->arglist() &&
             !context->trailer()[0]->arglist()->argument().empty() &&
             xacc::container::contains(
                 bufferNames,
                 context->trailer()[0]->arglist()->argument(0)->getText()))) {
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
        } else {
          if (!context->trailer().empty()) {
            // A classical call-like expression: i.e. not a kernel call:
            // Just output it *as-is* to the C++ stream.
            // We can hook more sophisticated code-gen here if required.
            // std::cout << "Callable: " << context->getText() << "\n";
            std::stringstream ss;

            if (context->trailer()[0]->arglist() &&
                !context->trailer()[0]->arglist()->argument().empty()) {
              const auto &argList =
                  context->trailer()[0]->arglist()->argument();
              ss << inst_name << "(";
              for (size_t i = 0; i < argList.size(); ++i) {                
                ss << rewriteFunctionArgument(*(argList[i]));                
                if (i != argList.size() - 1) {
                  ss << ", ";
                }
              }
              ss << ");\n";
            } else {
              ss << context->getText() << ";\n";
            }
            result.first = ss.str();
          }
        }
      }
    }
    return 0;
  }

  antlrcpp::Any visitFor_stmt(pyxasmParser::For_stmtContext *context) override {
    // Rewrite:
    // Python: "for <var> in <expr>:"
    // C++: for (auto var: <expr>) {}
    // Note: we add range(int) as a C++ function to support this common pattern.
    // or
    // Python: "for <idx>,<var> in enumerate(<listvar>):"
    // C++: for (auto [idx, var] : enumerate(listvar))
    auto iter_container = context->testlist()->test()[0]->getText();
    std::string counter_expr = context->exprlist()->expr()[0]->getText();
    // Add the for loop variable to the tracking list as well.
    new_var = counter_expr;
    if (context->exprlist()->expr().size() > 1) {
      counter_expr = "[" + counter_expr;
      for (int i = 1; i < context->exprlist()->expr().size(); i++) {
        counter_expr += ", " + context->exprlist()->expr()[i]->getText();
      }
      counter_expr += "]";
    }

    std::stringstream ss;
    ss << "for (auto " << counter_expr << " : " << iter_container << ") {\n";
    result.first = ss.str();
    in_for_loop = true;
    return 0;
  }

  antlrcpp::Any visitExpr_stmt(pyxasmParser::Expr_stmtContext *ctx) override {
    if (ctx->ASSIGN().size() == 1 && ctx->testlist_star_expr().size() == 2) {
      // Handle simple assignment: a = expr
      std::stringstream ss;
      const std::string lhs = ctx->testlist_star_expr(0)->getText();
      std::string rhs = replacePythonConstants(
          replaceMeasureAssignment(ctx->testlist_star_expr(1)->getText()));

      if (lhs.find(",") != std::string::npos) {
        // this is
        // var1, var2, ... = some_tuple_thing
        // We only support var1, var2 = ... for now
        // where ... is a pair-like object
        std::vector<std::string> suffix{".first", ".second"};
        auto vars = xacc::split(lhs, ',');
        for (auto [i, var] : qcor::enumerate(vars)) {
          if (xacc::container::contains(declared_var_names, var)) {
            ss << var << " = " << rhs << suffix[i] << ";\n";
          } else {
            ss << "auto " << var << " = " << rhs << suffix[i] << ";\n";
            new_var = lhs;
          }
        }
      } else {
        // Strategy: try to traverse the rhs to see if there is a possible rewrite;
        // Otherwise, use the text as is.
        is_processing_sub_expr = true;
        // clear the sub_node_translation  
        sub_node_translation.str(std::string());

        // visit arg sub-node:
        visitChildren(ctx->testlist_star_expr(1));

        // Check if there is a rewrite:
        if (!sub_node_translation.str().empty()) {
          // Update RHS
          rhs = replacePythonConstants(
              replaceMeasureAssignment(sub_node_translation.str()));
        }

        if (xacc::container::contains(declared_var_names, lhs)) {
          ss << lhs << " = " << rhs << "; \n";
        } else {
          // New variable: need to add *auto*
          ss << "auto " << lhs << " = " << rhs << "; \n";
          new_var = lhs;
        }
      }

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

  virtual antlrcpp::Any visitIf_stmt(
      pyxasmParser::If_stmtContext *ctx) override {
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

  // A helper to rewrite function argument by traversing the node to see
  // if there is a potential rewrite.
  // Use case: inline expressions
  // e.g. X(q[0:3])
  // slicing of the qreg 'q' then call the broadcast X op.
  // i.e., we need to rewrite the arg to q.extract_range({0, 3}).
  std::string
  rewriteFunctionArgument(pyxasmParser::ArgumentContext &in_argContext) {
    // Strategy: try to traverse the argument context to see if there is a
    // possible rewrite; i.e. it may be another atom_expression that we have a
    // handler for. Otherwise, use the text as is.
    // We need this flag to prevent parsing quantum instructions as sub-expressions.
    // e.g. QCOR operators (X, Y, Z) in an observable definition shouldn't be 
    // processed as instructions.
    is_processing_sub_expr = true;
    // clear the sub_node_translation
    sub_node_translation.str(std::string());

    // visit arg sub-node:
    visitChildren(&in_argContext);

    // Check if there is a rewrite:
    if (!sub_node_translation.str().empty()) {
      // Update RHS
      return sub_node_translation.str();
    }
    // Returns the string as is
    return in_argContext.getText();
  }
};
#include "pyxasm_token_collector.hpp"

#include <iostream>
#include <memory>
#include <set>

#include "clang/Basic/TokenKinds.h"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "pyxasmLexer.h"
#include "pyxasmParser.h"
#include "pyxasm_visitor.hpp"
#include "qrt_mapper.hpp"

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL PyXasmTokenCollectorActivator : public BundleActivator {
 public:
  PyXasmTokenCollectorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::PyXasmTokenCollector>();
    context.RegisterService<qcor::TokenCollector>(xt);
    // context.RegisterService<xacc::OptionsProvider>(acc);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

}  // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(PyXasmTokenCollectorActivator)

namespace qcor {

void PyXasmTokenCollector::collect(clang::Preprocessor &PP,
                                   clang::CachedTokens &Toks,
                                   std::vector<std::string> bufferNames,
                                   std::stringstream &ss, const std::string &kernel_name) {
  // NEW STRATEGY
  // Loop over tokens, get source file info like line number and
  // indentation. Construct vector of lines, and for each line
  // we will call the antlr parser to map to C++ code

  auto &sm = PP.getSourceManager();
  std::vector<std::pair<std::string, int>> lines;
  std::string line = "";
  auto current_line_number = sm.getSpellingLineNumber(Toks[0].getLocation());
  int last_col_number = 0;
  for (int i = 0; i < Toks.size(); i++) {
    auto location = Toks[i].getLocation();
    auto col_number = sm.getSpellingColumnNumber(location);
    auto line_number = sm.getSpellingLineNumber(location);
    if (current_line_number != line_number) {
      lines.push_back({line, col_number});
      line = "";
      current_line_number = line_number;
    }
    // std::cout << "LINE/COL: " << line_number << ", " << col_number << "\n";
    line += PP.getSpelling(Toks[i]);
    if (Toks[i].is(clang::tok::TokenKind::kw_for)) {
      // Right now we only assume 'for var in range'
      // or 'for i , var in enumerate( list_var ):
      // Slurp all elements of the for loop stmt, separate with spaces
      std::string for_stmt = " "; i++;
      while(Toks[i].isNot(clang::tok::TokenKind::colon)) {
        for_stmt += PP.getSpelling(Toks[i]) + " ";
        i++;
      }
      line += for_stmt;
    }

    // If statement or while statement:
    // Add a space b/w tokens.
    // Note: Python has an "elif" token, which doesn't have a C++ equiv.
    if (Toks[i].is(clang::tok::TokenKind::kw_if) ||
        PP.getSpelling(Toks[i]) == "elif" ||
        Toks[i].is(clang::tok::TokenKind::kw_while)) {
      line += " ";
      i += 1;
      line += PP.getSpelling(Toks[i]);
    }

    last_col_number = col_number;
  }

  // add last line
  lines.push_back({line, last_col_number});

  using namespace antlr4;

  int previous_col = lines[0].second;
  int line_counter = 0;
  // Add all the kernel args to the list of *known* arguments.
  // i.e. when we see an assignment expression where this arg. is the LHS,
  // we don't add *auto * to the codegen.
  std::vector<std::string> local_vars = [&]() -> std::vector<std::string> {
    if (::quantum::kernels_in_translation_unit.empty()) {
      return {};
    }
    const std::string this_kernel_name =
        kernel_name.empty() ? ::quantum::kernels_in_translation_unit.back()
                            : kernel_name;
    const auto &[arg_types, arg_names] =
        ::quantum::kernel_signatures_in_translation_unit[this_kernel_name];
    return arg_names;
  }();
  // Tracking the Python scopes by the indent of code blocks
  std::stack<int> scope_block_indent;
  for (const auto &line : lines) {
    // std::cout << "processing line " << line_counter << " of " << lines.size()
    //           << ": " << line.first << ", " << line.second << std::boolalpha
    //           << ", " << !scope_block_indent.empty() << "\n";

    pyxasm_visitor visitor(bufferNames, local_vars);
    // Should we close a 'for'/'if' scope after this statement
    // If > 0, indicate the number of for blocks to be closed.
    int nb_closing_scopes = 0;
    // If the stack is not empty and this line changed column to an outside
    // scope:
    while (!scope_block_indent.empty() &&
           line.second < scope_block_indent.top()) {
      // Pop the stack and flag to close the scope afterward
      scope_block_indent.pop();
      nb_closing_scopes++;
    }

    std::string lineText = line.first;
    // Enter a new for scope block (for/if/etc.) -> push to the stack
    // Note: we rewrite Python if .. elif .. else as follows:
    // Python:
    // if (cond1):
    //   code1
    // elif (cond2):
    //   code2
    // else:
    //   code3
    // ===============
    // C++:
    // if (cond1) {
    //   code1
    // }
    // else if (cond2) {
    //   code2
    // }
    // else {
    //   code3
    // }

    if (line.first.find("for ") != std::string::npos ||
        // Starts with 'if'
        line.first.rfind("if ", 0) == 0) {
      scope_block_indent.push(line.second);
    } else if (line.first == "else:") {
      ss << "else {\n";
      scope_block_indent.push(line.second);
    }
    // Starts with 'elif'
    else if (line.first.rfind("elif ", 0) == 0) {
      // Rewrite it to
      // else if () { }
      ss << "else ";
      scope_block_indent.push(line.second);
      // Remove the first two characters ("el")
      // hence this line will be parsed as an idependent C++ if block:
      lineText.erase(0, 2);
    } else if (line.first.rfind("while ", 0) == 0) {
      // rewrite to 
      // while (condition) {}
      // Just capture the indent level to close the scope properly
      scope_block_indent.push(line.second);
    }
    // is_in_for_loop = line.first.find("for ") != std::string::npos &&
    // line.second >= previous_col;

    ANTLRInputStream input(lineText);
    pyxasmLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    pyxasmParser parser(&tokens);

    lexer.removeErrorListeners();
    parser.removeErrorListeners();

    tree::ParseTree *tree = parser.single_input();

    visitor.visitChildren(tree);

    if (visitor.result.second) {
      // this was an xacc instruction
      qcor::qrt_mapper qrt_visitor;
      visitor.result.second->accept(xacc::as_shared_ptr(&qrt_visitor));
      ss << qrt_visitor.get_new_src();
    } else {
      // this was a classical code string
      ss << visitor.result.first;
    }

    if (nb_closing_scopes > 0) {
      // std::cout << "Close " << nb_closing_scopes << " for scopes.\n";
      // need to close out the c++ or loop
      for (int i = 0; i < nb_closing_scopes; ++i) {
        ss << "}\n";
      }
    }
    previous_col = line.second;
    line_counter++;
    if (!visitor.new_var.empty()) {
      // A new local variable was declared, add to the tracking list.
      local_vars.emplace_back(visitor.new_var);
    }
  }
  // If there are open scope blocks here,
  // e.g. for loops at the end of the function body.
  while (!scope_block_indent.empty()) {
    scope_block_indent.pop();
    ss << "}\n";
  }
}
}  // namespace qcor
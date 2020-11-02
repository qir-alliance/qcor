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
                                   std::stringstream &ss) {
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
    // std::cout << PP.getSpelling(Toks[i]) << "\n";
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
      line += " ";
      // add var space
      i += 1;
      line += PP.getSpelling(Toks[i]);
      line += " ";
      // add 'in space'
      i += 1;
      line += PP.getSpelling(Toks[i]);
      line += " ";
    }
    last_col_number = col_number;
  }

  // add last line
  lines.push_back({line, last_col_number});

  using namespace antlr4;

  int previous_col = lines[0].second;
  int line_counter = 0;
  // Tracking the scope of for loops by their indent
  std::stack<int> for_loop_indent;
  for (const auto &line : lines) {
    // std::cout << "processing line " << line_counter << " of " << lines.size()
    //           << ": " << line.first << ", " << line.second << std::boolalpha
    //           << ", " << !for_loop_indent.empty() << "\n";

    pyxasm_visitor visitor(bufferNames);
    // Should we close a 'for' scope after this statement
    // If > 0, indicate the number of for blocks to be closed.
    int close_for_scopes = 0;
    // If the stack is not empty and this line changed column to an outside
    // scope:
    while (!for_loop_indent.empty() && line.second < for_loop_indent.top()) {
      // Pop the stack and flag to close the scope afterward
      for_loop_indent.pop();
      close_for_scopes++;
    }

    // Enter a new for loop -> push to the stack
    if (line.first.find("for ") != std::string::npos) {
      for_loop_indent.push(line.second);
    }

    // is_in_for_loop = line.first.find("for ") != std::string::npos &&
    // line.second >= previous_col;

    ANTLRInputStream input(line.first);
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

    if (close_for_scopes > 0) {
      // std::cout << "Close " << close_for_scopes << " for scopes.\n";
      // need to close out the c++ or loop
      for (int i = 0; i < close_for_scopes; ++i) {
        ss << "}\n";
      }
    }
    previous_col = line.second;
    line_counter++;
  }
  // If there are open for scope blocks here,
  // i.e. for loops at the end of the function body.
  while (!for_loop_indent.empty()) {
    for_loop_indent.pop();
    ss << "}\n";
  }
}
}  // namespace qcor
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
  line += PP.getSpelling(Toks[0]);
  int last_col_number = 0;
  for (int i = 1; i < Toks.size(); i++) {
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
  bool is_in_for_loop = false;
  int line_counter = 0;
  for (const auto &line : lines) {
    // std::cout << "processing line " << line_counter << " of " << lines.size()
    //           << ": " << line.first << ", " << line.second << std::boolalpha
    //           << ", " << is_in_for_loop << "\n";

    pyxasm_visitor visitor;

    if (line.first.find("for ") != std::string::npos) {
      is_in_for_loop = true;
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

    if ((is_in_for_loop && line.second < previous_col) ||
        (is_in_for_loop && line_counter == lines.size() - 1)) {
      // we are now not in a for loop...
      is_in_for_loop = false;
      // need to close out the c++ or loop
      ss << "}\n";
    }

    previous_col = line.second;
    line_counter++;
  }

}
}  // namespace qcor
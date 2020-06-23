#include "xasm_token_collector.hpp"

#include "qrt_mapper.hpp"
#include "xasm_singleLexer.h"
#include "xasm_singleParser.h"
#include "xasm_single_visitor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "clang/Basic/TokenKinds.h"

#include <iostream>
#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL XasmTokenCollectorActivator : public BundleActivator {

public:
  XasmTokenCollectorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::XasmTokenCollector>();
    context.RegisterService<qcor::TokenCollector>(xt);
    // context.RegisterService<xacc::OptionsProvider>(acc);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(XasmTokenCollectorActivator)

namespace qcor {

void XasmTokenCollector::collect(clang::Preprocessor &PP,
                                 clang::CachedTokens &Toks,
                                 std::vector<std::string> bufferNames,
                                 std::stringstream &ss) {

  // NEW STRATEGY
  // Implement to split Toks into lines / stmts
  // and then use the xasm_single antlr visitor / parser
  // to translate that to IR OR classical code strings,
  // and then map IR to QRT calls and write to ss.

  // Each line should be terminated with semicolon OR
  // a control flow statement with an optional ending l_brace OR
  // a r_brace ending the control flow block

  std::vector<std::string> lines;
  int i = 0;
  while (true) {
    if (i >= Toks.size())
      break;

    auto current_token = Toks[i];
    if (current_token.is(clang::tok::kw_for)) {
      std::stringstream for_line;
      // slurp up the for loop to the end of the r_paren
      for_line << PP.getSpelling(current_token) << " ";
      i++;
      int l_paren_count = 0;
      while (true) {
        current_token = Toks[i];
        if (current_token.is(clang::tok::l_paren)) {
          l_paren_count++;
          for_line << PP.getSpelling(current_token) << " ";
        } else if (current_token.is(clang::tok::r_paren)) {
          for_line << PP.getSpelling(current_token) << " ";
          l_paren_count--;
          if (l_paren_count == 0) {
            // we reached the end, slurp up optional l_brace
            if (Toks[i + 1].is(clang::tok::l_brace)) {
              for_line << "{\n";
              i++;
            }

            break;
          }
        } else {
          for_line << PP.getSpelling(current_token) << " ";
        }
        i++;
      }

      lines.push_back(for_line.str());
    } else if (current_token.is(clang::tok::kw_if)) {
      // fill out if stmt here
    } else if (current_token.is(clang::tok::r_brace)) {

      lines.push_back("}\n");

    } else {

      // here we have some general statement, so
      // search til the semi colon
      std::stringstream ss;
      while (true) {
        // if we have ::, dont want to add it as : :
        std::string space = " ";
        if (current_token.is(clang::tok::colon) &&
            Toks[i + 1].is(clang::tok::colon)) {
          space = "";
        }

        ss << PP.getSpelling(current_token) << space;
        i++;
        current_token = Toks[i];
        if (current_token.is(clang::tok::semi)) {
          ss << ";";
          break;
        }
      }

      lines.push_back(ss.str());
    }

    i++;
  }

  // Loop over all lines, and parse them
  // with the custom single-line xasm parser.
  // this will produce either classical code strings
  // or quantum IR from xacc.
  using namespace antlr4;
  for (const auto &line : lines) {
    // xasm_single_result_type result;
    xasm_single_visitor visitor; //(result);

    ANTLRInputStream input(line);
    xasm_singleLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    xasm_singleParser parser(&tokens);

    tree::ParseTree *tree = parser.line();

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
  }
}
} // namespace qcor
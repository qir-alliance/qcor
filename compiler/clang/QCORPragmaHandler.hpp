#ifndef COMPILER_CLANG_QCORPRAGMAHANDLER_HPP__
#define COMPILER_CLANG_QCORPRAGMAHANDLER_HPP__
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/TokenKinds.def"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include <iostream>
#include <sstream>

using namespace clang;

namespace qcor {
namespace compiler {
class QCORPragmaHandler : public PragmaHandler {
protected:
  Rewriter& rewriter;
public:
  QCORPragmaHandler(Rewriter& r) : PragmaHandler("qcor"), rewriter(r) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                    Token &FirstTok) override {
        std::cout << "QCOR HANDLING PRAGMA:\n";
        Token Tok;
  // first slurp the directive content in a string.
  std::stringstream ss;
  SmallVector<Token, 16> Pragmas;
  int found = -1;
  auto sl = FirstTok.getLocation();
  sl.dump(PP.getSourceManager());

  std::string declaration;
//   FirstTok.getAnnotationRange().dump(PP.getSourceManager());
  while(true) {
    PP.Lex(Tok);
    if (Tok.is(tok::r_brace) && found == 0) {
        auto end = Tok.getLocation();
        sl.dump(PP.getSourceManager());
        end.dump(PP.getSourceManager());
        rewriter.ReplaceText(SourceRange(sl, end), "{}\n");
        PP.EnterToken(Tok);

        break;
    }

    if (Tok.is(tok::l_brace)) {
        if (found == -1) {
            declaration = ss.str();
            ss = std::stringstream();
        }
        found++;
    }
    if (Tok.is(tok::r_brace)) {
        found--;
    }

    // if(Tok.isNot(tok::eod))
      ss << PP.getSpelling(Tok);
  }

  std::cout << "declaration: " << declaration << "\n";
    std::cout << "body: " << ss.str() << "\n";

//   Tok.startToken();
// //   Tok.setKind(tok::annotannot_pragma_unused);//annot_pragma_my_annotate);
//   Tok.setLocation(FirstTok.getLocation());
//   Tok.setAnnotationEndLoc(FirstTok.getLocation());
//   // there should be something better that this strdup :-/
//   Tok.setAnnotationValue(strdup(ss.str().c_str()));

//   PP.EnterToken(Tok);

                //   SourceLocation PragmaLocation = Tok.getLocation();
                //   PragmaLocation.dump(PP.getSourceManager());
                //   Tok.getEndLoc().dump(PP.getSourceManager());
    //    std::cout << "LOCATION: " << PragmaLocation.

                    }
};
} // namespace compiler
} // namespace qcor

#endif
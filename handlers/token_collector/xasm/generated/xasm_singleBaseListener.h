
// Generated from xasm_single.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "xasm_singleListener.h"


namespace xasm {

/**
 * This class provides an empty implementation of xasm_singleListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  xasm_singleBaseListener : public xasm_singleListener {
public:

  virtual void enterLine(xasm_singleParser::LineContext * /*ctx*/) override { }
  virtual void exitLine(xasm_singleParser::LineContext * /*ctx*/) override { }

  virtual void enterStatement(xasm_singleParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(xasm_singleParser::StatementContext * /*ctx*/) override { }

  virtual void enterComment(xasm_singleParser::CommentContext * /*ctx*/) override { }
  virtual void exitComment(xasm_singleParser::CommentContext * /*ctx*/) override { }

  virtual void enterQinst(xasm_singleParser::QinstContext * /*ctx*/) override { }
  virtual void exitQinst(xasm_singleParser::QinstContext * /*ctx*/) override { }

  virtual void enterCinst(xasm_singleParser::CinstContext * /*ctx*/) override { }
  virtual void exitCinst(xasm_singleParser::CinstContext * /*ctx*/) override { }

  virtual void enterCpp_type(xasm_singleParser::Cpp_typeContext * /*ctx*/) override { }
  virtual void exitCpp_type(xasm_singleParser::Cpp_typeContext * /*ctx*/) override { }

  virtual void enterCompare(xasm_singleParser::CompareContext * /*ctx*/) override { }
  virtual void exitCompare(xasm_singleParser::CompareContext * /*ctx*/) override { }

  virtual void enterExplist(xasm_singleParser::ExplistContext * /*ctx*/) override { }
  virtual void exitExplist(xasm_singleParser::ExplistContext * /*ctx*/) override { }

  virtual void enterExp(xasm_singleParser::ExpContext * /*ctx*/) override { }
  virtual void exitExp(xasm_singleParser::ExpContext * /*ctx*/) override { }

  virtual void enterUnaryop(xasm_singleParser::UnaryopContext * /*ctx*/) override { }
  virtual void exitUnaryop(xasm_singleParser::UnaryopContext * /*ctx*/) override { }

  virtual void enterId(xasm_singleParser::IdContext * /*ctx*/) override { }
  virtual void exitId(xasm_singleParser::IdContext * /*ctx*/) override { }

  virtual void enterReal(xasm_singleParser::RealContext * /*ctx*/) override { }
  virtual void exitReal(xasm_singleParser::RealContext * /*ctx*/) override { }

  virtual void enterString(xasm_singleParser::StringContext * /*ctx*/) override { }
  virtual void exitString(xasm_singleParser::StringContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

}  // namespace xasm

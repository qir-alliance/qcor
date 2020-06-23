
// Generated from xasm_single.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "xasm_singleParser.h"


namespace xasm {

/**
 * This interface defines an abstract listener for a parse tree produced by xasm_singleParser.
 */
class  xasm_singleListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterLine(xasm_singleParser::LineContext *ctx) = 0;
  virtual void exitLine(xasm_singleParser::LineContext *ctx) = 0;

  virtual void enterStatement(xasm_singleParser::StatementContext *ctx) = 0;
  virtual void exitStatement(xasm_singleParser::StatementContext *ctx) = 0;

  virtual void enterComment(xasm_singleParser::CommentContext *ctx) = 0;
  virtual void exitComment(xasm_singleParser::CommentContext *ctx) = 0;

  virtual void enterQinst(xasm_singleParser::QinstContext *ctx) = 0;
  virtual void exitQinst(xasm_singleParser::QinstContext *ctx) = 0;

  virtual void enterCinst(xasm_singleParser::CinstContext *ctx) = 0;
  virtual void exitCinst(xasm_singleParser::CinstContext *ctx) = 0;

  virtual void enterCpp_type(xasm_singleParser::Cpp_typeContext *ctx) = 0;
  virtual void exitCpp_type(xasm_singleParser::Cpp_typeContext *ctx) = 0;

  virtual void enterCompare(xasm_singleParser::CompareContext *ctx) = 0;
  virtual void exitCompare(xasm_singleParser::CompareContext *ctx) = 0;

  virtual void enterExplist(xasm_singleParser::ExplistContext *ctx) = 0;
  virtual void exitExplist(xasm_singleParser::ExplistContext *ctx) = 0;

  virtual void enterExp(xasm_singleParser::ExpContext *ctx) = 0;
  virtual void exitExp(xasm_singleParser::ExpContext *ctx) = 0;

  virtual void enterUnaryop(xasm_singleParser::UnaryopContext *ctx) = 0;
  virtual void exitUnaryop(xasm_singleParser::UnaryopContext *ctx) = 0;

  virtual void enterId(xasm_singleParser::IdContext *ctx) = 0;
  virtual void exitId(xasm_singleParser::IdContext *ctx) = 0;

  virtual void enterReal(xasm_singleParser::RealContext *ctx) = 0;
  virtual void exitReal(xasm_singleParser::RealContext *ctx) = 0;

  virtual void enterString(xasm_singleParser::StringContext *ctx) = 0;
  virtual void exitString(xasm_singleParser::StringContext *ctx) = 0;


};

}  // namespace xasm


// Generated from xasm_single.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "xasm_singleParser.h"


namespace xasm {

/**
 * This class defines an abstract visitor for a parse tree
 * produced by xasm_singleParser.
 */
class  xasm_singleVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by xasm_singleParser.
   */
    virtual antlrcpp::Any visitLine(xasm_singleParser::LineContext *context) = 0;

    virtual antlrcpp::Any visitStatement(xasm_singleParser::StatementContext *context) = 0;

    virtual antlrcpp::Any visitComment(xasm_singleParser::CommentContext *context) = 0;

    virtual antlrcpp::Any visitQinst(xasm_singleParser::QinstContext *context) = 0;

    virtual antlrcpp::Any visitCinst(xasm_singleParser::CinstContext *context) = 0;

    virtual antlrcpp::Any visitCpp_type(xasm_singleParser::Cpp_typeContext *context) = 0;

    virtual antlrcpp::Any visitCompare(xasm_singleParser::CompareContext *context) = 0;

    virtual antlrcpp::Any visitExplist(xasm_singleParser::ExplistContext *context) = 0;

    virtual antlrcpp::Any visitExp(xasm_singleParser::ExpContext *context) = 0;

    virtual antlrcpp::Any visitUnaryop(xasm_singleParser::UnaryopContext *context) = 0;

    virtual antlrcpp::Any visitId(xasm_singleParser::IdContext *context) = 0;

    virtual antlrcpp::Any visitReal(xasm_singleParser::RealContext *context) = 0;

    virtual antlrcpp::Any visitString(xasm_singleParser::StringContext *context) = 0;


};

}  // namespace xasm

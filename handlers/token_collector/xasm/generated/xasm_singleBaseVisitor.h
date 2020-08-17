
// Generated from xasm_single.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "xasm_singleVisitor.h"


namespace xasm {

/**
 * This class provides an empty implementation of xasm_singleVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  xasm_singleBaseVisitor : public xasm_singleVisitor {
public:

  virtual antlrcpp::Any visitLine(xasm_singleParser::LineContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStatement(xasm_singleParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitComment(xasm_singleParser::CommentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQinst(xasm_singleParser::QinstContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCinst(xasm_singleParser::CinstContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCpp_type(xasm_singleParser::Cpp_typeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCompare(xasm_singleParser::CompareContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExplist(xasm_singleParser::ExplistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExp(xasm_singleParser::ExpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnaryop(xasm_singleParser::UnaryopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitId(xasm_singleParser::IdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReal(xasm_singleParser::RealContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitString(xasm_singleParser::StringContext *ctx) override {
    return visitChildren(ctx);
  }


};

}  // namespace xasm

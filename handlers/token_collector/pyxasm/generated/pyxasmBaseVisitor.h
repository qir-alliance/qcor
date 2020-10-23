
// Generated from pyxasm.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "pyxasmVisitor.h"


namespace pyxasm {

/**
 * This class provides an empty implementation of pyxasmVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  pyxasmBaseVisitor : public pyxasmVisitor {
public:

  virtual antlrcpp::Any visitSingle_input(pyxasmParser::Single_inputContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFile_input(pyxasmParser::File_inputContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitEval_input(pyxasmParser::Eval_inputContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDecorator(pyxasmParser::DecoratorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDecorators(pyxasmParser::DecoratorsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDecorated(pyxasmParser::DecoratedContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAsync_funcdef(pyxasmParser::Async_funcdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFuncdef(pyxasmParser::FuncdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitParameters(pyxasmParser::ParametersContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTypedargslist(pyxasmParser::TypedargslistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTfpdef(pyxasmParser::TfpdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVarargslist(pyxasmParser::VarargslistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVfpdef(pyxasmParser::VfpdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStmt(pyxasmParser::StmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSimple_stmt(pyxasmParser::Simple_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSmall_stmt(pyxasmParser::Small_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpr_stmt(pyxasmParser::Expr_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAnnassign(pyxasmParser::AnnassignContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTestlist_star_expr(pyxasmParser::Testlist_star_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAugassign(pyxasmParser::AugassignContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDel_stmt(pyxasmParser::Del_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPass_stmt(pyxasmParser::Pass_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFlow_stmt(pyxasmParser::Flow_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBreak_stmt(pyxasmParser::Break_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitContinue_stmt(pyxasmParser::Continue_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReturn_stmt(pyxasmParser::Return_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitYield_stmt(pyxasmParser::Yield_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitRaise_stmt(pyxasmParser::Raise_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitImport_stmt(pyxasmParser::Import_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitImport_name(pyxasmParser::Import_nameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitImport_from(pyxasmParser::Import_fromContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitImport_as_name(pyxasmParser::Import_as_nameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDotted_as_name(pyxasmParser::Dotted_as_nameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitImport_as_names(pyxasmParser::Import_as_namesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDotted_as_names(pyxasmParser::Dotted_as_namesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDotted_name(pyxasmParser::Dotted_nameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitGlobal_stmt(pyxasmParser::Global_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNonlocal_stmt(pyxasmParser::Nonlocal_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssert_stmt(pyxasmParser::Assert_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCompound_stmt(pyxasmParser::Compound_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAsync_stmt(pyxasmParser::Async_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIf_stmt(pyxasmParser::If_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWhile_stmt(pyxasmParser::While_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFor_stmt(pyxasmParser::For_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTry_stmt(pyxasmParser::Try_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWith_stmt(pyxasmParser::With_stmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWith_item(pyxasmParser::With_itemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExcept_clause(pyxasmParser::Except_clauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSuite(pyxasmParser::SuiteContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTest(pyxasmParser::TestContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTest_nocond(pyxasmParser::Test_nocondContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLambdef(pyxasmParser::LambdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLambdef_nocond(pyxasmParser::Lambdef_nocondContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitOr_test(pyxasmParser::Or_testContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAnd_test(pyxasmParser::And_testContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNot_test(pyxasmParser::Not_testContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitComparison(pyxasmParser::ComparisonContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitComp_op(pyxasmParser::Comp_opContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStar_expr(pyxasmParser::Star_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpr(pyxasmParser::ExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitXor_expr(pyxasmParser::Xor_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAnd_expr(pyxasmParser::And_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitShift_expr(pyxasmParser::Shift_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArith_expr(pyxasmParser::Arith_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTerm(pyxasmParser::TermContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFactor(pyxasmParser::FactorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPower(pyxasmParser::PowerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAtom_expr(pyxasmParser::Atom_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAtom(pyxasmParser::AtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTestlist_comp(pyxasmParser::Testlist_compContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTrailer(pyxasmParser::TrailerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSubscriptlist(pyxasmParser::SubscriptlistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSubscript(pyxasmParser::SubscriptContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSliceop(pyxasmParser::SliceopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExprlist(pyxasmParser::ExprlistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTestlist(pyxasmParser::TestlistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDictorsetmaker(pyxasmParser::DictorsetmakerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassdef(pyxasmParser::ClassdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArglist(pyxasmParser::ArglistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArgument(pyxasmParser::ArgumentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitComp_iter(pyxasmParser::Comp_iterContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitComp_for(pyxasmParser::Comp_forContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitComp_if(pyxasmParser::Comp_ifContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitEncoding_decl(pyxasmParser::Encoding_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitYield_expr(pyxasmParser::Yield_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitYield_arg(pyxasmParser::Yield_argContext *ctx) override {
    return visitChildren(ctx);
  }


};

}  // namespace pyxasm

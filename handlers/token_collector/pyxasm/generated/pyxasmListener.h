
// Generated from pyxasm.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "pyxasmParser.h"


namespace pyxasm {

/**
 * This interface defines an abstract listener for a parse tree produced by pyxasmParser.
 */
class  pyxasmListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterSingle_input(pyxasmParser::Single_inputContext *ctx) = 0;
  virtual void exitSingle_input(pyxasmParser::Single_inputContext *ctx) = 0;

  virtual void enterFile_input(pyxasmParser::File_inputContext *ctx) = 0;
  virtual void exitFile_input(pyxasmParser::File_inputContext *ctx) = 0;

  virtual void enterEval_input(pyxasmParser::Eval_inputContext *ctx) = 0;
  virtual void exitEval_input(pyxasmParser::Eval_inputContext *ctx) = 0;

  virtual void enterDecorator(pyxasmParser::DecoratorContext *ctx) = 0;
  virtual void exitDecorator(pyxasmParser::DecoratorContext *ctx) = 0;

  virtual void enterDecorators(pyxasmParser::DecoratorsContext *ctx) = 0;
  virtual void exitDecorators(pyxasmParser::DecoratorsContext *ctx) = 0;

  virtual void enterDecorated(pyxasmParser::DecoratedContext *ctx) = 0;
  virtual void exitDecorated(pyxasmParser::DecoratedContext *ctx) = 0;

  virtual void enterAsync_funcdef(pyxasmParser::Async_funcdefContext *ctx) = 0;
  virtual void exitAsync_funcdef(pyxasmParser::Async_funcdefContext *ctx) = 0;

  virtual void enterFuncdef(pyxasmParser::FuncdefContext *ctx) = 0;
  virtual void exitFuncdef(pyxasmParser::FuncdefContext *ctx) = 0;

  virtual void enterParameters(pyxasmParser::ParametersContext *ctx) = 0;
  virtual void exitParameters(pyxasmParser::ParametersContext *ctx) = 0;

  virtual void enterTypedargslist(pyxasmParser::TypedargslistContext *ctx) = 0;
  virtual void exitTypedargslist(pyxasmParser::TypedargslistContext *ctx) = 0;

  virtual void enterTfpdef(pyxasmParser::TfpdefContext *ctx) = 0;
  virtual void exitTfpdef(pyxasmParser::TfpdefContext *ctx) = 0;

  virtual void enterVarargslist(pyxasmParser::VarargslistContext *ctx) = 0;
  virtual void exitVarargslist(pyxasmParser::VarargslistContext *ctx) = 0;

  virtual void enterVfpdef(pyxasmParser::VfpdefContext *ctx) = 0;
  virtual void exitVfpdef(pyxasmParser::VfpdefContext *ctx) = 0;

  virtual void enterStmt(pyxasmParser::StmtContext *ctx) = 0;
  virtual void exitStmt(pyxasmParser::StmtContext *ctx) = 0;

  virtual void enterSimple_stmt(pyxasmParser::Simple_stmtContext *ctx) = 0;
  virtual void exitSimple_stmt(pyxasmParser::Simple_stmtContext *ctx) = 0;

  virtual void enterSmall_stmt(pyxasmParser::Small_stmtContext *ctx) = 0;
  virtual void exitSmall_stmt(pyxasmParser::Small_stmtContext *ctx) = 0;

  virtual void enterExpr_stmt(pyxasmParser::Expr_stmtContext *ctx) = 0;
  virtual void exitExpr_stmt(pyxasmParser::Expr_stmtContext *ctx) = 0;

  virtual void enterAnnassign(pyxasmParser::AnnassignContext *ctx) = 0;
  virtual void exitAnnassign(pyxasmParser::AnnassignContext *ctx) = 0;

  virtual void enterTestlist_star_expr(pyxasmParser::Testlist_star_exprContext *ctx) = 0;
  virtual void exitTestlist_star_expr(pyxasmParser::Testlist_star_exprContext *ctx) = 0;

  virtual void enterAugassign(pyxasmParser::AugassignContext *ctx) = 0;
  virtual void exitAugassign(pyxasmParser::AugassignContext *ctx) = 0;

  virtual void enterDel_stmt(pyxasmParser::Del_stmtContext *ctx) = 0;
  virtual void exitDel_stmt(pyxasmParser::Del_stmtContext *ctx) = 0;

  virtual void enterPass_stmt(pyxasmParser::Pass_stmtContext *ctx) = 0;
  virtual void exitPass_stmt(pyxasmParser::Pass_stmtContext *ctx) = 0;

  virtual void enterFlow_stmt(pyxasmParser::Flow_stmtContext *ctx) = 0;
  virtual void exitFlow_stmt(pyxasmParser::Flow_stmtContext *ctx) = 0;

  virtual void enterBreak_stmt(pyxasmParser::Break_stmtContext *ctx) = 0;
  virtual void exitBreak_stmt(pyxasmParser::Break_stmtContext *ctx) = 0;

  virtual void enterContinue_stmt(pyxasmParser::Continue_stmtContext *ctx) = 0;
  virtual void exitContinue_stmt(pyxasmParser::Continue_stmtContext *ctx) = 0;

  virtual void enterReturn_stmt(pyxasmParser::Return_stmtContext *ctx) = 0;
  virtual void exitReturn_stmt(pyxasmParser::Return_stmtContext *ctx) = 0;

  virtual void enterYield_stmt(pyxasmParser::Yield_stmtContext *ctx) = 0;
  virtual void exitYield_stmt(pyxasmParser::Yield_stmtContext *ctx) = 0;

  virtual void enterRaise_stmt(pyxasmParser::Raise_stmtContext *ctx) = 0;
  virtual void exitRaise_stmt(pyxasmParser::Raise_stmtContext *ctx) = 0;

  virtual void enterImport_stmt(pyxasmParser::Import_stmtContext *ctx) = 0;
  virtual void exitImport_stmt(pyxasmParser::Import_stmtContext *ctx) = 0;

  virtual void enterImport_name(pyxasmParser::Import_nameContext *ctx) = 0;
  virtual void exitImport_name(pyxasmParser::Import_nameContext *ctx) = 0;

  virtual void enterImport_from(pyxasmParser::Import_fromContext *ctx) = 0;
  virtual void exitImport_from(pyxasmParser::Import_fromContext *ctx) = 0;

  virtual void enterImport_as_name(pyxasmParser::Import_as_nameContext *ctx) = 0;
  virtual void exitImport_as_name(pyxasmParser::Import_as_nameContext *ctx) = 0;

  virtual void enterDotted_as_name(pyxasmParser::Dotted_as_nameContext *ctx) = 0;
  virtual void exitDotted_as_name(pyxasmParser::Dotted_as_nameContext *ctx) = 0;

  virtual void enterImport_as_names(pyxasmParser::Import_as_namesContext *ctx) = 0;
  virtual void exitImport_as_names(pyxasmParser::Import_as_namesContext *ctx) = 0;

  virtual void enterDotted_as_names(pyxasmParser::Dotted_as_namesContext *ctx) = 0;
  virtual void exitDotted_as_names(pyxasmParser::Dotted_as_namesContext *ctx) = 0;

  virtual void enterDotted_name(pyxasmParser::Dotted_nameContext *ctx) = 0;
  virtual void exitDotted_name(pyxasmParser::Dotted_nameContext *ctx) = 0;

  virtual void enterGlobal_stmt(pyxasmParser::Global_stmtContext *ctx) = 0;
  virtual void exitGlobal_stmt(pyxasmParser::Global_stmtContext *ctx) = 0;

  virtual void enterNonlocal_stmt(pyxasmParser::Nonlocal_stmtContext *ctx) = 0;
  virtual void exitNonlocal_stmt(pyxasmParser::Nonlocal_stmtContext *ctx) = 0;

  virtual void enterAssert_stmt(pyxasmParser::Assert_stmtContext *ctx) = 0;
  virtual void exitAssert_stmt(pyxasmParser::Assert_stmtContext *ctx) = 0;

  virtual void enterCompound_stmt(pyxasmParser::Compound_stmtContext *ctx) = 0;
  virtual void exitCompound_stmt(pyxasmParser::Compound_stmtContext *ctx) = 0;

  virtual void enterAsync_stmt(pyxasmParser::Async_stmtContext *ctx) = 0;
  virtual void exitAsync_stmt(pyxasmParser::Async_stmtContext *ctx) = 0;

  virtual void enterIf_stmt(pyxasmParser::If_stmtContext *ctx) = 0;
  virtual void exitIf_stmt(pyxasmParser::If_stmtContext *ctx) = 0;

  virtual void enterWhile_stmt(pyxasmParser::While_stmtContext *ctx) = 0;
  virtual void exitWhile_stmt(pyxasmParser::While_stmtContext *ctx) = 0;

  virtual void enterFor_stmt(pyxasmParser::For_stmtContext *ctx) = 0;
  virtual void exitFor_stmt(pyxasmParser::For_stmtContext *ctx) = 0;

  virtual void enterTry_stmt(pyxasmParser::Try_stmtContext *ctx) = 0;
  virtual void exitTry_stmt(pyxasmParser::Try_stmtContext *ctx) = 0;

  virtual void enterWith_stmt(pyxasmParser::With_stmtContext *ctx) = 0;
  virtual void exitWith_stmt(pyxasmParser::With_stmtContext *ctx) = 0;

  virtual void enterWith_item(pyxasmParser::With_itemContext *ctx) = 0;
  virtual void exitWith_item(pyxasmParser::With_itemContext *ctx) = 0;

  virtual void enterExcept_clause(pyxasmParser::Except_clauseContext *ctx) = 0;
  virtual void exitExcept_clause(pyxasmParser::Except_clauseContext *ctx) = 0;

  virtual void enterSuite(pyxasmParser::SuiteContext *ctx) = 0;
  virtual void exitSuite(pyxasmParser::SuiteContext *ctx) = 0;

  virtual void enterTest(pyxasmParser::TestContext *ctx) = 0;
  virtual void exitTest(pyxasmParser::TestContext *ctx) = 0;

  virtual void enterTest_nocond(pyxasmParser::Test_nocondContext *ctx) = 0;
  virtual void exitTest_nocond(pyxasmParser::Test_nocondContext *ctx) = 0;

  virtual void enterLambdef(pyxasmParser::LambdefContext *ctx) = 0;
  virtual void exitLambdef(pyxasmParser::LambdefContext *ctx) = 0;

  virtual void enterLambdef_nocond(pyxasmParser::Lambdef_nocondContext *ctx) = 0;
  virtual void exitLambdef_nocond(pyxasmParser::Lambdef_nocondContext *ctx) = 0;

  virtual void enterOr_test(pyxasmParser::Or_testContext *ctx) = 0;
  virtual void exitOr_test(pyxasmParser::Or_testContext *ctx) = 0;

  virtual void enterAnd_test(pyxasmParser::And_testContext *ctx) = 0;
  virtual void exitAnd_test(pyxasmParser::And_testContext *ctx) = 0;

  virtual void enterNot_test(pyxasmParser::Not_testContext *ctx) = 0;
  virtual void exitNot_test(pyxasmParser::Not_testContext *ctx) = 0;

  virtual void enterComparison(pyxasmParser::ComparisonContext *ctx) = 0;
  virtual void exitComparison(pyxasmParser::ComparisonContext *ctx) = 0;

  virtual void enterComp_op(pyxasmParser::Comp_opContext *ctx) = 0;
  virtual void exitComp_op(pyxasmParser::Comp_opContext *ctx) = 0;

  virtual void enterStar_expr(pyxasmParser::Star_exprContext *ctx) = 0;
  virtual void exitStar_expr(pyxasmParser::Star_exprContext *ctx) = 0;

  virtual void enterExpr(pyxasmParser::ExprContext *ctx) = 0;
  virtual void exitExpr(pyxasmParser::ExprContext *ctx) = 0;

  virtual void enterXor_expr(pyxasmParser::Xor_exprContext *ctx) = 0;
  virtual void exitXor_expr(pyxasmParser::Xor_exprContext *ctx) = 0;

  virtual void enterAnd_expr(pyxasmParser::And_exprContext *ctx) = 0;
  virtual void exitAnd_expr(pyxasmParser::And_exprContext *ctx) = 0;

  virtual void enterShift_expr(pyxasmParser::Shift_exprContext *ctx) = 0;
  virtual void exitShift_expr(pyxasmParser::Shift_exprContext *ctx) = 0;

  virtual void enterArith_expr(pyxasmParser::Arith_exprContext *ctx) = 0;
  virtual void exitArith_expr(pyxasmParser::Arith_exprContext *ctx) = 0;

  virtual void enterTerm(pyxasmParser::TermContext *ctx) = 0;
  virtual void exitTerm(pyxasmParser::TermContext *ctx) = 0;

  virtual void enterFactor(pyxasmParser::FactorContext *ctx) = 0;
  virtual void exitFactor(pyxasmParser::FactorContext *ctx) = 0;

  virtual void enterPower(pyxasmParser::PowerContext *ctx) = 0;
  virtual void exitPower(pyxasmParser::PowerContext *ctx) = 0;

  virtual void enterAtom_expr(pyxasmParser::Atom_exprContext *ctx) = 0;
  virtual void exitAtom_expr(pyxasmParser::Atom_exprContext *ctx) = 0;

  virtual void enterAtom(pyxasmParser::AtomContext *ctx) = 0;
  virtual void exitAtom(pyxasmParser::AtomContext *ctx) = 0;

  virtual void enterTestlist_comp(pyxasmParser::Testlist_compContext *ctx) = 0;
  virtual void exitTestlist_comp(pyxasmParser::Testlist_compContext *ctx) = 0;

  virtual void enterTrailer(pyxasmParser::TrailerContext *ctx) = 0;
  virtual void exitTrailer(pyxasmParser::TrailerContext *ctx) = 0;

  virtual void enterSubscriptlist(pyxasmParser::SubscriptlistContext *ctx) = 0;
  virtual void exitSubscriptlist(pyxasmParser::SubscriptlistContext *ctx) = 0;

  virtual void enterSubscript(pyxasmParser::SubscriptContext *ctx) = 0;
  virtual void exitSubscript(pyxasmParser::SubscriptContext *ctx) = 0;

  virtual void enterSliceop(pyxasmParser::SliceopContext *ctx) = 0;
  virtual void exitSliceop(pyxasmParser::SliceopContext *ctx) = 0;

  virtual void enterExprlist(pyxasmParser::ExprlistContext *ctx) = 0;
  virtual void exitExprlist(pyxasmParser::ExprlistContext *ctx) = 0;

  virtual void enterTestlist(pyxasmParser::TestlistContext *ctx) = 0;
  virtual void exitTestlist(pyxasmParser::TestlistContext *ctx) = 0;

  virtual void enterDictorsetmaker(pyxasmParser::DictorsetmakerContext *ctx) = 0;
  virtual void exitDictorsetmaker(pyxasmParser::DictorsetmakerContext *ctx) = 0;

  virtual void enterClassdef(pyxasmParser::ClassdefContext *ctx) = 0;
  virtual void exitClassdef(pyxasmParser::ClassdefContext *ctx) = 0;

  virtual void enterArglist(pyxasmParser::ArglistContext *ctx) = 0;
  virtual void exitArglist(pyxasmParser::ArglistContext *ctx) = 0;

  virtual void enterArgument(pyxasmParser::ArgumentContext *ctx) = 0;
  virtual void exitArgument(pyxasmParser::ArgumentContext *ctx) = 0;

  virtual void enterComp_iter(pyxasmParser::Comp_iterContext *ctx) = 0;
  virtual void exitComp_iter(pyxasmParser::Comp_iterContext *ctx) = 0;

  virtual void enterComp_for(pyxasmParser::Comp_forContext *ctx) = 0;
  virtual void exitComp_for(pyxasmParser::Comp_forContext *ctx) = 0;

  virtual void enterComp_if(pyxasmParser::Comp_ifContext *ctx) = 0;
  virtual void exitComp_if(pyxasmParser::Comp_ifContext *ctx) = 0;

  virtual void enterEncoding_decl(pyxasmParser::Encoding_declContext *ctx) = 0;
  virtual void exitEncoding_decl(pyxasmParser::Encoding_declContext *ctx) = 0;

  virtual void enterYield_expr(pyxasmParser::Yield_exprContext *ctx) = 0;
  virtual void exitYield_expr(pyxasmParser::Yield_exprContext *ctx) = 0;

  virtual void enterYield_arg(pyxasmParser::Yield_argContext *ctx) = 0;
  virtual void exitYield_arg(pyxasmParser::Yield_argContext *ctx) = 0;


};

}  // namespace pyxasm

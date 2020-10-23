
// Generated from pyxasm.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "pyxasmListener.h"


namespace pyxasm {

/**
 * This class provides an empty implementation of pyxasmListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  pyxasmBaseListener : public pyxasmListener {
public:

  virtual void enterSingle_input(pyxasmParser::Single_inputContext * /*ctx*/) override { }
  virtual void exitSingle_input(pyxasmParser::Single_inputContext * /*ctx*/) override { }

  virtual void enterFile_input(pyxasmParser::File_inputContext * /*ctx*/) override { }
  virtual void exitFile_input(pyxasmParser::File_inputContext * /*ctx*/) override { }

  virtual void enterEval_input(pyxasmParser::Eval_inputContext * /*ctx*/) override { }
  virtual void exitEval_input(pyxasmParser::Eval_inputContext * /*ctx*/) override { }

  virtual void enterDecorator(pyxasmParser::DecoratorContext * /*ctx*/) override { }
  virtual void exitDecorator(pyxasmParser::DecoratorContext * /*ctx*/) override { }

  virtual void enterDecorators(pyxasmParser::DecoratorsContext * /*ctx*/) override { }
  virtual void exitDecorators(pyxasmParser::DecoratorsContext * /*ctx*/) override { }

  virtual void enterDecorated(pyxasmParser::DecoratedContext * /*ctx*/) override { }
  virtual void exitDecorated(pyxasmParser::DecoratedContext * /*ctx*/) override { }

  virtual void enterAsync_funcdef(pyxasmParser::Async_funcdefContext * /*ctx*/) override { }
  virtual void exitAsync_funcdef(pyxasmParser::Async_funcdefContext * /*ctx*/) override { }

  virtual void enterFuncdef(pyxasmParser::FuncdefContext * /*ctx*/) override { }
  virtual void exitFuncdef(pyxasmParser::FuncdefContext * /*ctx*/) override { }

  virtual void enterParameters(pyxasmParser::ParametersContext * /*ctx*/) override { }
  virtual void exitParameters(pyxasmParser::ParametersContext * /*ctx*/) override { }

  virtual void enterTypedargslist(pyxasmParser::TypedargslistContext * /*ctx*/) override { }
  virtual void exitTypedargslist(pyxasmParser::TypedargslistContext * /*ctx*/) override { }

  virtual void enterTfpdef(pyxasmParser::TfpdefContext * /*ctx*/) override { }
  virtual void exitTfpdef(pyxasmParser::TfpdefContext * /*ctx*/) override { }

  virtual void enterVarargslist(pyxasmParser::VarargslistContext * /*ctx*/) override { }
  virtual void exitVarargslist(pyxasmParser::VarargslistContext * /*ctx*/) override { }

  virtual void enterVfpdef(pyxasmParser::VfpdefContext * /*ctx*/) override { }
  virtual void exitVfpdef(pyxasmParser::VfpdefContext * /*ctx*/) override { }

  virtual void enterStmt(pyxasmParser::StmtContext * /*ctx*/) override { }
  virtual void exitStmt(pyxasmParser::StmtContext * /*ctx*/) override { }

  virtual void enterSimple_stmt(pyxasmParser::Simple_stmtContext * /*ctx*/) override { }
  virtual void exitSimple_stmt(pyxasmParser::Simple_stmtContext * /*ctx*/) override { }

  virtual void enterSmall_stmt(pyxasmParser::Small_stmtContext * /*ctx*/) override { }
  virtual void exitSmall_stmt(pyxasmParser::Small_stmtContext * /*ctx*/) override { }

  virtual void enterExpr_stmt(pyxasmParser::Expr_stmtContext * /*ctx*/) override { }
  virtual void exitExpr_stmt(pyxasmParser::Expr_stmtContext * /*ctx*/) override { }

  virtual void enterAnnassign(pyxasmParser::AnnassignContext * /*ctx*/) override { }
  virtual void exitAnnassign(pyxasmParser::AnnassignContext * /*ctx*/) override { }

  virtual void enterTestlist_star_expr(pyxasmParser::Testlist_star_exprContext * /*ctx*/) override { }
  virtual void exitTestlist_star_expr(pyxasmParser::Testlist_star_exprContext * /*ctx*/) override { }

  virtual void enterAugassign(pyxasmParser::AugassignContext * /*ctx*/) override { }
  virtual void exitAugassign(pyxasmParser::AugassignContext * /*ctx*/) override { }

  virtual void enterDel_stmt(pyxasmParser::Del_stmtContext * /*ctx*/) override { }
  virtual void exitDel_stmt(pyxasmParser::Del_stmtContext * /*ctx*/) override { }

  virtual void enterPass_stmt(pyxasmParser::Pass_stmtContext * /*ctx*/) override { }
  virtual void exitPass_stmt(pyxasmParser::Pass_stmtContext * /*ctx*/) override { }

  virtual void enterFlow_stmt(pyxasmParser::Flow_stmtContext * /*ctx*/) override { }
  virtual void exitFlow_stmt(pyxasmParser::Flow_stmtContext * /*ctx*/) override { }

  virtual void enterBreak_stmt(pyxasmParser::Break_stmtContext * /*ctx*/) override { }
  virtual void exitBreak_stmt(pyxasmParser::Break_stmtContext * /*ctx*/) override { }

  virtual void enterContinue_stmt(pyxasmParser::Continue_stmtContext * /*ctx*/) override { }
  virtual void exitContinue_stmt(pyxasmParser::Continue_stmtContext * /*ctx*/) override { }

  virtual void enterReturn_stmt(pyxasmParser::Return_stmtContext * /*ctx*/) override { }
  virtual void exitReturn_stmt(pyxasmParser::Return_stmtContext * /*ctx*/) override { }

  virtual void enterYield_stmt(pyxasmParser::Yield_stmtContext * /*ctx*/) override { }
  virtual void exitYield_stmt(pyxasmParser::Yield_stmtContext * /*ctx*/) override { }

  virtual void enterRaise_stmt(pyxasmParser::Raise_stmtContext * /*ctx*/) override { }
  virtual void exitRaise_stmt(pyxasmParser::Raise_stmtContext * /*ctx*/) override { }

  virtual void enterImport_stmt(pyxasmParser::Import_stmtContext * /*ctx*/) override { }
  virtual void exitImport_stmt(pyxasmParser::Import_stmtContext * /*ctx*/) override { }

  virtual void enterImport_name(pyxasmParser::Import_nameContext * /*ctx*/) override { }
  virtual void exitImport_name(pyxasmParser::Import_nameContext * /*ctx*/) override { }

  virtual void enterImport_from(pyxasmParser::Import_fromContext * /*ctx*/) override { }
  virtual void exitImport_from(pyxasmParser::Import_fromContext * /*ctx*/) override { }

  virtual void enterImport_as_name(pyxasmParser::Import_as_nameContext * /*ctx*/) override { }
  virtual void exitImport_as_name(pyxasmParser::Import_as_nameContext * /*ctx*/) override { }

  virtual void enterDotted_as_name(pyxasmParser::Dotted_as_nameContext * /*ctx*/) override { }
  virtual void exitDotted_as_name(pyxasmParser::Dotted_as_nameContext * /*ctx*/) override { }

  virtual void enterImport_as_names(pyxasmParser::Import_as_namesContext * /*ctx*/) override { }
  virtual void exitImport_as_names(pyxasmParser::Import_as_namesContext * /*ctx*/) override { }

  virtual void enterDotted_as_names(pyxasmParser::Dotted_as_namesContext * /*ctx*/) override { }
  virtual void exitDotted_as_names(pyxasmParser::Dotted_as_namesContext * /*ctx*/) override { }

  virtual void enterDotted_name(pyxasmParser::Dotted_nameContext * /*ctx*/) override { }
  virtual void exitDotted_name(pyxasmParser::Dotted_nameContext * /*ctx*/) override { }

  virtual void enterGlobal_stmt(pyxasmParser::Global_stmtContext * /*ctx*/) override { }
  virtual void exitGlobal_stmt(pyxasmParser::Global_stmtContext * /*ctx*/) override { }

  virtual void enterNonlocal_stmt(pyxasmParser::Nonlocal_stmtContext * /*ctx*/) override { }
  virtual void exitNonlocal_stmt(pyxasmParser::Nonlocal_stmtContext * /*ctx*/) override { }

  virtual void enterAssert_stmt(pyxasmParser::Assert_stmtContext * /*ctx*/) override { }
  virtual void exitAssert_stmt(pyxasmParser::Assert_stmtContext * /*ctx*/) override { }

  virtual void enterCompound_stmt(pyxasmParser::Compound_stmtContext * /*ctx*/) override { }
  virtual void exitCompound_stmt(pyxasmParser::Compound_stmtContext * /*ctx*/) override { }

  virtual void enterAsync_stmt(pyxasmParser::Async_stmtContext * /*ctx*/) override { }
  virtual void exitAsync_stmt(pyxasmParser::Async_stmtContext * /*ctx*/) override { }

  virtual void enterIf_stmt(pyxasmParser::If_stmtContext * /*ctx*/) override { }
  virtual void exitIf_stmt(pyxasmParser::If_stmtContext * /*ctx*/) override { }

  virtual void enterWhile_stmt(pyxasmParser::While_stmtContext * /*ctx*/) override { }
  virtual void exitWhile_stmt(pyxasmParser::While_stmtContext * /*ctx*/) override { }

  virtual void enterFor_stmt(pyxasmParser::For_stmtContext * /*ctx*/) override { }
  virtual void exitFor_stmt(pyxasmParser::For_stmtContext * /*ctx*/) override { }

  virtual void enterTry_stmt(pyxasmParser::Try_stmtContext * /*ctx*/) override { }
  virtual void exitTry_stmt(pyxasmParser::Try_stmtContext * /*ctx*/) override { }

  virtual void enterWith_stmt(pyxasmParser::With_stmtContext * /*ctx*/) override { }
  virtual void exitWith_stmt(pyxasmParser::With_stmtContext * /*ctx*/) override { }

  virtual void enterWith_item(pyxasmParser::With_itemContext * /*ctx*/) override { }
  virtual void exitWith_item(pyxasmParser::With_itemContext * /*ctx*/) override { }

  virtual void enterExcept_clause(pyxasmParser::Except_clauseContext * /*ctx*/) override { }
  virtual void exitExcept_clause(pyxasmParser::Except_clauseContext * /*ctx*/) override { }

  virtual void enterSuite(pyxasmParser::SuiteContext * /*ctx*/) override { }
  virtual void exitSuite(pyxasmParser::SuiteContext * /*ctx*/) override { }

  virtual void enterTest(pyxasmParser::TestContext * /*ctx*/) override { }
  virtual void exitTest(pyxasmParser::TestContext * /*ctx*/) override { }

  virtual void enterTest_nocond(pyxasmParser::Test_nocondContext * /*ctx*/) override { }
  virtual void exitTest_nocond(pyxasmParser::Test_nocondContext * /*ctx*/) override { }

  virtual void enterLambdef(pyxasmParser::LambdefContext * /*ctx*/) override { }
  virtual void exitLambdef(pyxasmParser::LambdefContext * /*ctx*/) override { }

  virtual void enterLambdef_nocond(pyxasmParser::Lambdef_nocondContext * /*ctx*/) override { }
  virtual void exitLambdef_nocond(pyxasmParser::Lambdef_nocondContext * /*ctx*/) override { }

  virtual void enterOr_test(pyxasmParser::Or_testContext * /*ctx*/) override { }
  virtual void exitOr_test(pyxasmParser::Or_testContext * /*ctx*/) override { }

  virtual void enterAnd_test(pyxasmParser::And_testContext * /*ctx*/) override { }
  virtual void exitAnd_test(pyxasmParser::And_testContext * /*ctx*/) override { }

  virtual void enterNot_test(pyxasmParser::Not_testContext * /*ctx*/) override { }
  virtual void exitNot_test(pyxasmParser::Not_testContext * /*ctx*/) override { }

  virtual void enterComparison(pyxasmParser::ComparisonContext * /*ctx*/) override { }
  virtual void exitComparison(pyxasmParser::ComparisonContext * /*ctx*/) override { }

  virtual void enterComp_op(pyxasmParser::Comp_opContext * /*ctx*/) override { }
  virtual void exitComp_op(pyxasmParser::Comp_opContext * /*ctx*/) override { }

  virtual void enterStar_expr(pyxasmParser::Star_exprContext * /*ctx*/) override { }
  virtual void exitStar_expr(pyxasmParser::Star_exprContext * /*ctx*/) override { }

  virtual void enterExpr(pyxasmParser::ExprContext * /*ctx*/) override { }
  virtual void exitExpr(pyxasmParser::ExprContext * /*ctx*/) override { }

  virtual void enterXor_expr(pyxasmParser::Xor_exprContext * /*ctx*/) override { }
  virtual void exitXor_expr(pyxasmParser::Xor_exprContext * /*ctx*/) override { }

  virtual void enterAnd_expr(pyxasmParser::And_exprContext * /*ctx*/) override { }
  virtual void exitAnd_expr(pyxasmParser::And_exprContext * /*ctx*/) override { }

  virtual void enterShift_expr(pyxasmParser::Shift_exprContext * /*ctx*/) override { }
  virtual void exitShift_expr(pyxasmParser::Shift_exprContext * /*ctx*/) override { }

  virtual void enterArith_expr(pyxasmParser::Arith_exprContext * /*ctx*/) override { }
  virtual void exitArith_expr(pyxasmParser::Arith_exprContext * /*ctx*/) override { }

  virtual void enterTerm(pyxasmParser::TermContext * /*ctx*/) override { }
  virtual void exitTerm(pyxasmParser::TermContext * /*ctx*/) override { }

  virtual void enterFactor(pyxasmParser::FactorContext * /*ctx*/) override { }
  virtual void exitFactor(pyxasmParser::FactorContext * /*ctx*/) override { }

  virtual void enterPower(pyxasmParser::PowerContext * /*ctx*/) override { }
  virtual void exitPower(pyxasmParser::PowerContext * /*ctx*/) override { }

  virtual void enterAtom_expr(pyxasmParser::Atom_exprContext * /*ctx*/) override { }
  virtual void exitAtom_expr(pyxasmParser::Atom_exprContext * /*ctx*/) override { }

  virtual void enterAtom(pyxasmParser::AtomContext * /*ctx*/) override { }
  virtual void exitAtom(pyxasmParser::AtomContext * /*ctx*/) override { }

  virtual void enterTestlist_comp(pyxasmParser::Testlist_compContext * /*ctx*/) override { }
  virtual void exitTestlist_comp(pyxasmParser::Testlist_compContext * /*ctx*/) override { }

  virtual void enterTrailer(pyxasmParser::TrailerContext * /*ctx*/) override { }
  virtual void exitTrailer(pyxasmParser::TrailerContext * /*ctx*/) override { }

  virtual void enterSubscriptlist(pyxasmParser::SubscriptlistContext * /*ctx*/) override { }
  virtual void exitSubscriptlist(pyxasmParser::SubscriptlistContext * /*ctx*/) override { }

  virtual void enterSubscript(pyxasmParser::SubscriptContext * /*ctx*/) override { }
  virtual void exitSubscript(pyxasmParser::SubscriptContext * /*ctx*/) override { }

  virtual void enterSliceop(pyxasmParser::SliceopContext * /*ctx*/) override { }
  virtual void exitSliceop(pyxasmParser::SliceopContext * /*ctx*/) override { }

  virtual void enterExprlist(pyxasmParser::ExprlistContext * /*ctx*/) override { }
  virtual void exitExprlist(pyxasmParser::ExprlistContext * /*ctx*/) override { }

  virtual void enterTestlist(pyxasmParser::TestlistContext * /*ctx*/) override { }
  virtual void exitTestlist(pyxasmParser::TestlistContext * /*ctx*/) override { }

  virtual void enterDictorsetmaker(pyxasmParser::DictorsetmakerContext * /*ctx*/) override { }
  virtual void exitDictorsetmaker(pyxasmParser::DictorsetmakerContext * /*ctx*/) override { }

  virtual void enterClassdef(pyxasmParser::ClassdefContext * /*ctx*/) override { }
  virtual void exitClassdef(pyxasmParser::ClassdefContext * /*ctx*/) override { }

  virtual void enterArglist(pyxasmParser::ArglistContext * /*ctx*/) override { }
  virtual void exitArglist(pyxasmParser::ArglistContext * /*ctx*/) override { }

  virtual void enterArgument(pyxasmParser::ArgumentContext * /*ctx*/) override { }
  virtual void exitArgument(pyxasmParser::ArgumentContext * /*ctx*/) override { }

  virtual void enterComp_iter(pyxasmParser::Comp_iterContext * /*ctx*/) override { }
  virtual void exitComp_iter(pyxasmParser::Comp_iterContext * /*ctx*/) override { }

  virtual void enterComp_for(pyxasmParser::Comp_forContext * /*ctx*/) override { }
  virtual void exitComp_for(pyxasmParser::Comp_forContext * /*ctx*/) override { }

  virtual void enterComp_if(pyxasmParser::Comp_ifContext * /*ctx*/) override { }
  virtual void exitComp_if(pyxasmParser::Comp_ifContext * /*ctx*/) override { }

  virtual void enterEncoding_decl(pyxasmParser::Encoding_declContext * /*ctx*/) override { }
  virtual void exitEncoding_decl(pyxasmParser::Encoding_declContext * /*ctx*/) override { }

  virtual void enterYield_expr(pyxasmParser::Yield_exprContext * /*ctx*/) override { }
  virtual void exitYield_expr(pyxasmParser::Yield_exprContext * /*ctx*/) override { }

  virtual void enterYield_arg(pyxasmParser::Yield_argContext * /*ctx*/) override { }
  virtual void exitYield_arg(pyxasmParser::Yield_argContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

}  // namespace pyxasm

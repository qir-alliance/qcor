
// Generated from pyxasm.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "pyxasmParser.h"


namespace pyxasm {

/**
 * This class defines an abstract visitor for a parse tree
 * produced by pyxasmParser.
 */
class  pyxasmVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by pyxasmParser.
   */
    virtual antlrcpp::Any visitSingle_input(pyxasmParser::Single_inputContext *context) = 0;

    virtual antlrcpp::Any visitFile_input(pyxasmParser::File_inputContext *context) = 0;

    virtual antlrcpp::Any visitEval_input(pyxasmParser::Eval_inputContext *context) = 0;

    virtual antlrcpp::Any visitDecorator(pyxasmParser::DecoratorContext *context) = 0;

    virtual antlrcpp::Any visitDecorators(pyxasmParser::DecoratorsContext *context) = 0;

    virtual antlrcpp::Any visitDecorated(pyxasmParser::DecoratedContext *context) = 0;

    virtual antlrcpp::Any visitAsync_funcdef(pyxasmParser::Async_funcdefContext *context) = 0;

    virtual antlrcpp::Any visitFuncdef(pyxasmParser::FuncdefContext *context) = 0;

    virtual antlrcpp::Any visitParameters(pyxasmParser::ParametersContext *context) = 0;

    virtual antlrcpp::Any visitTypedargslist(pyxasmParser::TypedargslistContext *context) = 0;

    virtual antlrcpp::Any visitTfpdef(pyxasmParser::TfpdefContext *context) = 0;

    virtual antlrcpp::Any visitVarargslist(pyxasmParser::VarargslistContext *context) = 0;

    virtual antlrcpp::Any visitVfpdef(pyxasmParser::VfpdefContext *context) = 0;

    virtual antlrcpp::Any visitStmt(pyxasmParser::StmtContext *context) = 0;

    virtual antlrcpp::Any visitSimple_stmt(pyxasmParser::Simple_stmtContext *context) = 0;

    virtual antlrcpp::Any visitSmall_stmt(pyxasmParser::Small_stmtContext *context) = 0;

    virtual antlrcpp::Any visitExpr_stmt(pyxasmParser::Expr_stmtContext *context) = 0;

    virtual antlrcpp::Any visitAnnassign(pyxasmParser::AnnassignContext *context) = 0;

    virtual antlrcpp::Any visitTestlist_star_expr(pyxasmParser::Testlist_star_exprContext *context) = 0;

    virtual antlrcpp::Any visitAugassign(pyxasmParser::AugassignContext *context) = 0;

    virtual antlrcpp::Any visitDel_stmt(pyxasmParser::Del_stmtContext *context) = 0;

    virtual antlrcpp::Any visitPass_stmt(pyxasmParser::Pass_stmtContext *context) = 0;

    virtual antlrcpp::Any visitFlow_stmt(pyxasmParser::Flow_stmtContext *context) = 0;

    virtual antlrcpp::Any visitBreak_stmt(pyxasmParser::Break_stmtContext *context) = 0;

    virtual antlrcpp::Any visitContinue_stmt(pyxasmParser::Continue_stmtContext *context) = 0;

    virtual antlrcpp::Any visitReturn_stmt(pyxasmParser::Return_stmtContext *context) = 0;

    virtual antlrcpp::Any visitYield_stmt(pyxasmParser::Yield_stmtContext *context) = 0;

    virtual antlrcpp::Any visitRaise_stmt(pyxasmParser::Raise_stmtContext *context) = 0;

    virtual antlrcpp::Any visitImport_stmt(pyxasmParser::Import_stmtContext *context) = 0;

    virtual antlrcpp::Any visitImport_name(pyxasmParser::Import_nameContext *context) = 0;

    virtual antlrcpp::Any visitImport_from(pyxasmParser::Import_fromContext *context) = 0;

    virtual antlrcpp::Any visitImport_as_name(pyxasmParser::Import_as_nameContext *context) = 0;

    virtual antlrcpp::Any visitDotted_as_name(pyxasmParser::Dotted_as_nameContext *context) = 0;

    virtual antlrcpp::Any visitImport_as_names(pyxasmParser::Import_as_namesContext *context) = 0;

    virtual antlrcpp::Any visitDotted_as_names(pyxasmParser::Dotted_as_namesContext *context) = 0;

    virtual antlrcpp::Any visitDotted_name(pyxasmParser::Dotted_nameContext *context) = 0;

    virtual antlrcpp::Any visitGlobal_stmt(pyxasmParser::Global_stmtContext *context) = 0;

    virtual antlrcpp::Any visitNonlocal_stmt(pyxasmParser::Nonlocal_stmtContext *context) = 0;

    virtual antlrcpp::Any visitAssert_stmt(pyxasmParser::Assert_stmtContext *context) = 0;

    virtual antlrcpp::Any visitCompound_stmt(pyxasmParser::Compound_stmtContext *context) = 0;

    virtual antlrcpp::Any visitAsync_stmt(pyxasmParser::Async_stmtContext *context) = 0;

    virtual antlrcpp::Any visitIf_stmt(pyxasmParser::If_stmtContext *context) = 0;

    virtual antlrcpp::Any visitWhile_stmt(pyxasmParser::While_stmtContext *context) = 0;

    virtual antlrcpp::Any visitFor_stmt(pyxasmParser::For_stmtContext *context) = 0;

    virtual antlrcpp::Any visitTry_stmt(pyxasmParser::Try_stmtContext *context) = 0;

    virtual antlrcpp::Any visitWith_stmt(pyxasmParser::With_stmtContext *context) = 0;

    virtual antlrcpp::Any visitWith_item(pyxasmParser::With_itemContext *context) = 0;

    virtual antlrcpp::Any visitExcept_clause(pyxasmParser::Except_clauseContext *context) = 0;

    virtual antlrcpp::Any visitSuite(pyxasmParser::SuiteContext *context) = 0;

    virtual antlrcpp::Any visitTest(pyxasmParser::TestContext *context) = 0;

    virtual antlrcpp::Any visitTest_nocond(pyxasmParser::Test_nocondContext *context) = 0;

    virtual antlrcpp::Any visitLambdef(pyxasmParser::LambdefContext *context) = 0;

    virtual antlrcpp::Any visitLambdef_nocond(pyxasmParser::Lambdef_nocondContext *context) = 0;

    virtual antlrcpp::Any visitOr_test(pyxasmParser::Or_testContext *context) = 0;

    virtual antlrcpp::Any visitAnd_test(pyxasmParser::And_testContext *context) = 0;

    virtual antlrcpp::Any visitNot_test(pyxasmParser::Not_testContext *context) = 0;

    virtual antlrcpp::Any visitComparison(pyxasmParser::ComparisonContext *context) = 0;

    virtual antlrcpp::Any visitComp_op(pyxasmParser::Comp_opContext *context) = 0;

    virtual antlrcpp::Any visitStar_expr(pyxasmParser::Star_exprContext *context) = 0;

    virtual antlrcpp::Any visitExpr(pyxasmParser::ExprContext *context) = 0;

    virtual antlrcpp::Any visitXor_expr(pyxasmParser::Xor_exprContext *context) = 0;

    virtual antlrcpp::Any visitAnd_expr(pyxasmParser::And_exprContext *context) = 0;

    virtual antlrcpp::Any visitShift_expr(pyxasmParser::Shift_exprContext *context) = 0;

    virtual antlrcpp::Any visitArith_expr(pyxasmParser::Arith_exprContext *context) = 0;

    virtual antlrcpp::Any visitTerm(pyxasmParser::TermContext *context) = 0;

    virtual antlrcpp::Any visitFactor(pyxasmParser::FactorContext *context) = 0;

    virtual antlrcpp::Any visitPower(pyxasmParser::PowerContext *context) = 0;

    virtual antlrcpp::Any visitAtom_expr(pyxasmParser::Atom_exprContext *context) = 0;

    virtual antlrcpp::Any visitAtom(pyxasmParser::AtomContext *context) = 0;

    virtual antlrcpp::Any visitTestlist_comp(pyxasmParser::Testlist_compContext *context) = 0;

    virtual antlrcpp::Any visitTrailer(pyxasmParser::TrailerContext *context) = 0;

    virtual antlrcpp::Any visitSubscriptlist(pyxasmParser::SubscriptlistContext *context) = 0;

    virtual antlrcpp::Any visitSubscript(pyxasmParser::SubscriptContext *context) = 0;

    virtual antlrcpp::Any visitSliceop(pyxasmParser::SliceopContext *context) = 0;

    virtual antlrcpp::Any visitExprlist(pyxasmParser::ExprlistContext *context) = 0;

    virtual antlrcpp::Any visitTestlist(pyxasmParser::TestlistContext *context) = 0;

    virtual antlrcpp::Any visitDictorsetmaker(pyxasmParser::DictorsetmakerContext *context) = 0;

    virtual antlrcpp::Any visitClassdef(pyxasmParser::ClassdefContext *context) = 0;

    virtual antlrcpp::Any visitArglist(pyxasmParser::ArglistContext *context) = 0;

    virtual antlrcpp::Any visitArgument(pyxasmParser::ArgumentContext *context) = 0;

    virtual antlrcpp::Any visitComp_iter(pyxasmParser::Comp_iterContext *context) = 0;

    virtual antlrcpp::Any visitComp_for(pyxasmParser::Comp_forContext *context) = 0;

    virtual antlrcpp::Any visitComp_if(pyxasmParser::Comp_ifContext *context) = 0;

    virtual antlrcpp::Any visitEncoding_decl(pyxasmParser::Encoding_declContext *context) = 0;

    virtual antlrcpp::Any visitYield_expr(pyxasmParser::Yield_exprContext *context) = 0;

    virtual antlrcpp::Any visitYield_arg(pyxasmParser::Yield_argContext *context) = 0;


};

}  // namespace pyxasm

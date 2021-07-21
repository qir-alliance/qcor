
// Generated from qasm3.g4 by ANTLR 4.9.2

#pragma once


#include "antlr4-runtime.h"
#include "qasm3Parser.h"


namespace qasm3 {

/**
 * This class defines an abstract visitor for a parse tree
 * produced by qasm3Parser.
 */
class  qasm3Visitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by qasm3Parser.
   */
    virtual antlrcpp::Any visitProgram(qasm3Parser::ProgramContext *context) = 0;

    virtual antlrcpp::Any visitHeader(qasm3Parser::HeaderContext *context) = 0;

    virtual antlrcpp::Any visitVersion(qasm3Parser::VersionContext *context) = 0;

    virtual antlrcpp::Any visitInclude(qasm3Parser::IncludeContext *context) = 0;

    virtual antlrcpp::Any visitGlobalStatement(qasm3Parser::GlobalStatementContext *context) = 0;

    virtual antlrcpp::Any visitStatement(qasm3Parser::StatementContext *context) = 0;

    virtual antlrcpp::Any visitCompute_action_stmt(qasm3Parser::Compute_action_stmtContext *context) = 0;

    virtual antlrcpp::Any visitQcor_test_statement(qasm3Parser::Qcor_test_statementContext *context) = 0;

    virtual antlrcpp::Any visitQuantumDeclarationStatement(qasm3Parser::QuantumDeclarationStatementContext *context) = 0;

    virtual antlrcpp::Any visitClassicalDeclarationStatement(qasm3Parser::ClassicalDeclarationStatementContext *context) = 0;

    virtual antlrcpp::Any visitClassicalAssignment(qasm3Parser::ClassicalAssignmentContext *context) = 0;

    virtual antlrcpp::Any visitAssignmentStatement(qasm3Parser::AssignmentStatementContext *context) = 0;

    virtual antlrcpp::Any visitReturnSignature(qasm3Parser::ReturnSignatureContext *context) = 0;

    virtual antlrcpp::Any visitDesignator(qasm3Parser::DesignatorContext *context) = 0;

    virtual antlrcpp::Any visitDoubleDesignator(qasm3Parser::DoubleDesignatorContext *context) = 0;

    virtual antlrcpp::Any visitIdentifierList(qasm3Parser::IdentifierListContext *context) = 0;

    virtual antlrcpp::Any visitAssociation(qasm3Parser::AssociationContext *context) = 0;

    virtual antlrcpp::Any visitQuantumType(qasm3Parser::QuantumTypeContext *context) = 0;

    virtual antlrcpp::Any visitQuantumDeclaration(qasm3Parser::QuantumDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitQuantumArgument(qasm3Parser::QuantumArgumentContext *context) = 0;

    virtual antlrcpp::Any visitQuantumArgumentList(qasm3Parser::QuantumArgumentListContext *context) = 0;

    virtual antlrcpp::Any visitBitType(qasm3Parser::BitTypeContext *context) = 0;

    virtual antlrcpp::Any visitSingleDesignatorType(qasm3Parser::SingleDesignatorTypeContext *context) = 0;

    virtual antlrcpp::Any visitDoubleDesignatorType(qasm3Parser::DoubleDesignatorTypeContext *context) = 0;

    virtual antlrcpp::Any visitNoDesignatorType(qasm3Parser::NoDesignatorTypeContext *context) = 0;

    virtual antlrcpp::Any visitClassicalType(qasm3Parser::ClassicalTypeContext *context) = 0;

    virtual antlrcpp::Any visitConstantDeclaration(qasm3Parser::ConstantDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitSingleDesignatorDeclaration(qasm3Parser::SingleDesignatorDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitDoubleDesignatorDeclaration(qasm3Parser::DoubleDesignatorDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitNoDesignatorDeclaration(qasm3Parser::NoDesignatorDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitBitDeclaration(qasm3Parser::BitDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitClassicalDeclaration(qasm3Parser::ClassicalDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitClassicalTypeList(qasm3Parser::ClassicalTypeListContext *context) = 0;

    virtual antlrcpp::Any visitClassicalArgument(qasm3Parser::ClassicalArgumentContext *context) = 0;

    virtual antlrcpp::Any visitClassicalArgumentList(qasm3Parser::ClassicalArgumentListContext *context) = 0;

    virtual antlrcpp::Any visitAliasStatement(qasm3Parser::AliasStatementContext *context) = 0;

    virtual antlrcpp::Any visitIndexIdentifier(qasm3Parser::IndexIdentifierContext *context) = 0;

    virtual antlrcpp::Any visitIndexIdentifierList(qasm3Parser::IndexIdentifierListContext *context) = 0;

    virtual antlrcpp::Any visitIndexEqualsAssignmentList(qasm3Parser::IndexEqualsAssignmentListContext *context) = 0;

    virtual antlrcpp::Any visitRangeDefinition(qasm3Parser::RangeDefinitionContext *context) = 0;

    virtual antlrcpp::Any visitQuantumGateDefinition(qasm3Parser::QuantumGateDefinitionContext *context) = 0;

    virtual antlrcpp::Any visitQuantumGateSignature(qasm3Parser::QuantumGateSignatureContext *context) = 0;

    virtual antlrcpp::Any visitQuantumBlock(qasm3Parser::QuantumBlockContext *context) = 0;

    virtual antlrcpp::Any visitQuantumLoop(qasm3Parser::QuantumLoopContext *context) = 0;

    virtual antlrcpp::Any visitQuantumLoopBlock(qasm3Parser::QuantumLoopBlockContext *context) = 0;

    virtual antlrcpp::Any visitQuantumStatement(qasm3Parser::QuantumStatementContext *context) = 0;

    virtual antlrcpp::Any visitQuantumInstruction(qasm3Parser::QuantumInstructionContext *context) = 0;

    virtual antlrcpp::Any visitQuantumPhase(qasm3Parser::QuantumPhaseContext *context) = 0;

    virtual antlrcpp::Any visitQuantumMeasurement(qasm3Parser::QuantumMeasurementContext *context) = 0;

    virtual antlrcpp::Any visitQuantumMeasurementAssignment(qasm3Parser::QuantumMeasurementAssignmentContext *context) = 0;

    virtual antlrcpp::Any visitQuantumBarrier(qasm3Parser::QuantumBarrierContext *context) = 0;

    virtual antlrcpp::Any visitQuantumGateModifier(qasm3Parser::QuantumGateModifierContext *context) = 0;

    virtual antlrcpp::Any visitQuantumGateCall(qasm3Parser::QuantumGateCallContext *context) = 0;

    virtual antlrcpp::Any visitQuantumGateName(qasm3Parser::QuantumGateNameContext *context) = 0;

    virtual antlrcpp::Any visitUnaryOperator(qasm3Parser::UnaryOperatorContext *context) = 0;

    virtual antlrcpp::Any visitRelationalOperator(qasm3Parser::RelationalOperatorContext *context) = 0;

    virtual antlrcpp::Any visitLogicalOperator(qasm3Parser::LogicalOperatorContext *context) = 0;

    virtual antlrcpp::Any visitExpressionStatement(qasm3Parser::ExpressionStatementContext *context) = 0;

    virtual antlrcpp::Any visitExpression(qasm3Parser::ExpressionContext *context) = 0;

    virtual antlrcpp::Any visitXOrExpression(qasm3Parser::XOrExpressionContext *context) = 0;

    virtual antlrcpp::Any visitBitAndExpression(qasm3Parser::BitAndExpressionContext *context) = 0;

    virtual antlrcpp::Any visitBitShiftExpression(qasm3Parser::BitShiftExpressionContext *context) = 0;

    virtual antlrcpp::Any visitAdditiveExpression(qasm3Parser::AdditiveExpressionContext *context) = 0;

    virtual antlrcpp::Any visitMultiplicativeExpression(qasm3Parser::MultiplicativeExpressionContext *context) = 0;

    virtual antlrcpp::Any visitUnaryExpression(qasm3Parser::UnaryExpressionContext *context) = 0;

    virtual antlrcpp::Any visitExpressionTerminator(qasm3Parser::ExpressionTerminatorContext *context) = 0;

    virtual antlrcpp::Any visitIncrementor(qasm3Parser::IncrementorContext *context) = 0;

    virtual antlrcpp::Any visitBuiltInCall(qasm3Parser::BuiltInCallContext *context) = 0;

    virtual antlrcpp::Any visitBuiltInMath(qasm3Parser::BuiltInMathContext *context) = 0;

    virtual antlrcpp::Any visitCastOperator(qasm3Parser::CastOperatorContext *context) = 0;

    virtual antlrcpp::Any visitExpressionList(qasm3Parser::ExpressionListContext *context) = 0;

    virtual antlrcpp::Any visitBooleanExpression(qasm3Parser::BooleanExpressionContext *context) = 0;

    virtual antlrcpp::Any visitComparsionExpression(qasm3Parser::ComparsionExpressionContext *context) = 0;

    virtual antlrcpp::Any visitEqualsExpression(qasm3Parser::EqualsExpressionContext *context) = 0;

    virtual antlrcpp::Any visitAssignmentOperator(qasm3Parser::AssignmentOperatorContext *context) = 0;

    virtual antlrcpp::Any visitEqualsAssignmentList(qasm3Parser::EqualsAssignmentListContext *context) = 0;

    virtual antlrcpp::Any visitMembershipTest(qasm3Parser::MembershipTestContext *context) = 0;

    virtual antlrcpp::Any visitSetDeclaration(qasm3Parser::SetDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitProgramBlock(qasm3Parser::ProgramBlockContext *context) = 0;

    virtual antlrcpp::Any visitBranchingStatement(qasm3Parser::BranchingStatementContext *context) = 0;

    virtual antlrcpp::Any visitLoopSignature(qasm3Parser::LoopSignatureContext *context) = 0;

    virtual antlrcpp::Any visitLoopStatement(qasm3Parser::LoopStatementContext *context) = 0;

    virtual antlrcpp::Any visitCLikeLoopStatement(qasm3Parser::CLikeLoopStatementContext *context) = 0;

    virtual antlrcpp::Any visitControlDirectiveStatement(qasm3Parser::ControlDirectiveStatementContext *context) = 0;

    virtual antlrcpp::Any visitControlDirective(qasm3Parser::ControlDirectiveContext *context) = 0;

    virtual antlrcpp::Any visitKernelDeclaration(qasm3Parser::KernelDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitKernelCall(qasm3Parser::KernelCallContext *context) = 0;

    virtual antlrcpp::Any visitSubroutineDefinition(qasm3Parser::SubroutineDefinitionContext *context) = 0;

    virtual antlrcpp::Any visitReturnStatement(qasm3Parser::ReturnStatementContext *context) = 0;

    virtual antlrcpp::Any visitSubroutineBlock(qasm3Parser::SubroutineBlockContext *context) = 0;

    virtual antlrcpp::Any visitSubroutineCall(qasm3Parser::SubroutineCallContext *context) = 0;

    virtual antlrcpp::Any visitPragma(qasm3Parser::PragmaContext *context) = 0;

    virtual antlrcpp::Any visitTimingType(qasm3Parser::TimingTypeContext *context) = 0;

    virtual antlrcpp::Any visitTimingBox(qasm3Parser::TimingBoxContext *context) = 0;

    virtual antlrcpp::Any visitTimingTerminator(qasm3Parser::TimingTerminatorContext *context) = 0;

    virtual antlrcpp::Any visitTimingIdentifier(qasm3Parser::TimingIdentifierContext *context) = 0;

    virtual antlrcpp::Any visitTimingInstructionName(qasm3Parser::TimingInstructionNameContext *context) = 0;

    virtual antlrcpp::Any visitTimingInstruction(qasm3Parser::TimingInstructionContext *context) = 0;

    virtual antlrcpp::Any visitTimingStatement(qasm3Parser::TimingStatementContext *context) = 0;

    virtual antlrcpp::Any visitCalibration(qasm3Parser::CalibrationContext *context) = 0;

    virtual antlrcpp::Any visitCalibrationGrammarDeclaration(qasm3Parser::CalibrationGrammarDeclarationContext *context) = 0;

    virtual antlrcpp::Any visitCalibrationDefinition(qasm3Parser::CalibrationDefinitionContext *context) = 0;

    virtual antlrcpp::Any visitCalibrationGrammar(qasm3Parser::CalibrationGrammarContext *context) = 0;

    virtual antlrcpp::Any visitCalibrationArgumentList(qasm3Parser::CalibrationArgumentListContext *context) = 0;


};

}  // namespace qasm3


// Generated from qasm3.g4 by ANTLR 4.9.2

#pragma once


#include "antlr4-runtime.h"
#include "qasm3Parser.h"


namespace qasm3 {

/**
 * This interface defines an abstract listener for a parse tree produced by qasm3Parser.
 */
class  qasm3Listener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterProgram(qasm3Parser::ProgramContext *ctx) = 0;
  virtual void exitProgram(qasm3Parser::ProgramContext *ctx) = 0;

  virtual void enterHeader(qasm3Parser::HeaderContext *ctx) = 0;
  virtual void exitHeader(qasm3Parser::HeaderContext *ctx) = 0;

  virtual void enterVersion(qasm3Parser::VersionContext *ctx) = 0;
  virtual void exitVersion(qasm3Parser::VersionContext *ctx) = 0;

  virtual void enterInclude(qasm3Parser::IncludeContext *ctx) = 0;
  virtual void exitInclude(qasm3Parser::IncludeContext *ctx) = 0;

  virtual void enterGlobalStatement(qasm3Parser::GlobalStatementContext *ctx) = 0;
  virtual void exitGlobalStatement(qasm3Parser::GlobalStatementContext *ctx) = 0;

  virtual void enterStatement(qasm3Parser::StatementContext *ctx) = 0;
  virtual void exitStatement(qasm3Parser::StatementContext *ctx) = 0;

  virtual void enterCompute_action_stmt(qasm3Parser::Compute_action_stmtContext *ctx) = 0;
  virtual void exitCompute_action_stmt(qasm3Parser::Compute_action_stmtContext *ctx) = 0;

  virtual void enterQcor_test_statement(qasm3Parser::Qcor_test_statementContext *ctx) = 0;
  virtual void exitQcor_test_statement(qasm3Parser::Qcor_test_statementContext *ctx) = 0;

  virtual void enterQuantumDeclarationStatement(qasm3Parser::QuantumDeclarationStatementContext *ctx) = 0;
  virtual void exitQuantumDeclarationStatement(qasm3Parser::QuantumDeclarationStatementContext *ctx) = 0;

  virtual void enterClassicalDeclarationStatement(qasm3Parser::ClassicalDeclarationStatementContext *ctx) = 0;
  virtual void exitClassicalDeclarationStatement(qasm3Parser::ClassicalDeclarationStatementContext *ctx) = 0;

  virtual void enterClassicalAssignment(qasm3Parser::ClassicalAssignmentContext *ctx) = 0;
  virtual void exitClassicalAssignment(qasm3Parser::ClassicalAssignmentContext *ctx) = 0;

  virtual void enterAssignmentStatement(qasm3Parser::AssignmentStatementContext *ctx) = 0;
  virtual void exitAssignmentStatement(qasm3Parser::AssignmentStatementContext *ctx) = 0;

  virtual void enterReturnSignature(qasm3Parser::ReturnSignatureContext *ctx) = 0;
  virtual void exitReturnSignature(qasm3Parser::ReturnSignatureContext *ctx) = 0;

  virtual void enterDesignator(qasm3Parser::DesignatorContext *ctx) = 0;
  virtual void exitDesignator(qasm3Parser::DesignatorContext *ctx) = 0;

  virtual void enterDoubleDesignator(qasm3Parser::DoubleDesignatorContext *ctx) = 0;
  virtual void exitDoubleDesignator(qasm3Parser::DoubleDesignatorContext *ctx) = 0;

  virtual void enterIdentifierList(qasm3Parser::IdentifierListContext *ctx) = 0;
  virtual void exitIdentifierList(qasm3Parser::IdentifierListContext *ctx) = 0;

  virtual void enterAssociation(qasm3Parser::AssociationContext *ctx) = 0;
  virtual void exitAssociation(qasm3Parser::AssociationContext *ctx) = 0;

  virtual void enterQuantumType(qasm3Parser::QuantumTypeContext *ctx) = 0;
  virtual void exitQuantumType(qasm3Parser::QuantumTypeContext *ctx) = 0;

  virtual void enterQuantumDeclaration(qasm3Parser::QuantumDeclarationContext *ctx) = 0;
  virtual void exitQuantumDeclaration(qasm3Parser::QuantumDeclarationContext *ctx) = 0;

  virtual void enterQuantumArgument(qasm3Parser::QuantumArgumentContext *ctx) = 0;
  virtual void exitQuantumArgument(qasm3Parser::QuantumArgumentContext *ctx) = 0;

  virtual void enterQuantumArgumentList(qasm3Parser::QuantumArgumentListContext *ctx) = 0;
  virtual void exitQuantumArgumentList(qasm3Parser::QuantumArgumentListContext *ctx) = 0;

  virtual void enterBitType(qasm3Parser::BitTypeContext *ctx) = 0;
  virtual void exitBitType(qasm3Parser::BitTypeContext *ctx) = 0;

  virtual void enterSingleDesignatorType(qasm3Parser::SingleDesignatorTypeContext *ctx) = 0;
  virtual void exitSingleDesignatorType(qasm3Parser::SingleDesignatorTypeContext *ctx) = 0;

  virtual void enterDoubleDesignatorType(qasm3Parser::DoubleDesignatorTypeContext *ctx) = 0;
  virtual void exitDoubleDesignatorType(qasm3Parser::DoubleDesignatorTypeContext *ctx) = 0;

  virtual void enterNoDesignatorType(qasm3Parser::NoDesignatorTypeContext *ctx) = 0;
  virtual void exitNoDesignatorType(qasm3Parser::NoDesignatorTypeContext *ctx) = 0;

  virtual void enterClassicalType(qasm3Parser::ClassicalTypeContext *ctx) = 0;
  virtual void exitClassicalType(qasm3Parser::ClassicalTypeContext *ctx) = 0;

  virtual void enterConstantDeclaration(qasm3Parser::ConstantDeclarationContext *ctx) = 0;
  virtual void exitConstantDeclaration(qasm3Parser::ConstantDeclarationContext *ctx) = 0;

  virtual void enterSingleDesignatorDeclaration(qasm3Parser::SingleDesignatorDeclarationContext *ctx) = 0;
  virtual void exitSingleDesignatorDeclaration(qasm3Parser::SingleDesignatorDeclarationContext *ctx) = 0;

  virtual void enterDoubleDesignatorDeclaration(qasm3Parser::DoubleDesignatorDeclarationContext *ctx) = 0;
  virtual void exitDoubleDesignatorDeclaration(qasm3Parser::DoubleDesignatorDeclarationContext *ctx) = 0;

  virtual void enterNoDesignatorDeclaration(qasm3Parser::NoDesignatorDeclarationContext *ctx) = 0;
  virtual void exitNoDesignatorDeclaration(qasm3Parser::NoDesignatorDeclarationContext *ctx) = 0;

  virtual void enterBitDeclaration(qasm3Parser::BitDeclarationContext *ctx) = 0;
  virtual void exitBitDeclaration(qasm3Parser::BitDeclarationContext *ctx) = 0;

  virtual void enterClassicalDeclaration(qasm3Parser::ClassicalDeclarationContext *ctx) = 0;
  virtual void exitClassicalDeclaration(qasm3Parser::ClassicalDeclarationContext *ctx) = 0;

  virtual void enterClassicalTypeList(qasm3Parser::ClassicalTypeListContext *ctx) = 0;
  virtual void exitClassicalTypeList(qasm3Parser::ClassicalTypeListContext *ctx) = 0;

  virtual void enterClassicalArgument(qasm3Parser::ClassicalArgumentContext *ctx) = 0;
  virtual void exitClassicalArgument(qasm3Parser::ClassicalArgumentContext *ctx) = 0;

  virtual void enterClassicalArgumentList(qasm3Parser::ClassicalArgumentListContext *ctx) = 0;
  virtual void exitClassicalArgumentList(qasm3Parser::ClassicalArgumentListContext *ctx) = 0;

  virtual void enterAliasStatement(qasm3Parser::AliasStatementContext *ctx) = 0;
  virtual void exitAliasStatement(qasm3Parser::AliasStatementContext *ctx) = 0;

  virtual void enterIndexIdentifier(qasm3Parser::IndexIdentifierContext *ctx) = 0;
  virtual void exitIndexIdentifier(qasm3Parser::IndexIdentifierContext *ctx) = 0;

  virtual void enterIndexIdentifierList(qasm3Parser::IndexIdentifierListContext *ctx) = 0;
  virtual void exitIndexIdentifierList(qasm3Parser::IndexIdentifierListContext *ctx) = 0;

  virtual void enterIndexEqualsAssignmentList(qasm3Parser::IndexEqualsAssignmentListContext *ctx) = 0;
  virtual void exitIndexEqualsAssignmentList(qasm3Parser::IndexEqualsAssignmentListContext *ctx) = 0;

  virtual void enterRangeDefinition(qasm3Parser::RangeDefinitionContext *ctx) = 0;
  virtual void exitRangeDefinition(qasm3Parser::RangeDefinitionContext *ctx) = 0;

  virtual void enterQuantumGateDefinition(qasm3Parser::QuantumGateDefinitionContext *ctx) = 0;
  virtual void exitQuantumGateDefinition(qasm3Parser::QuantumGateDefinitionContext *ctx) = 0;

  virtual void enterQuantumGateSignature(qasm3Parser::QuantumGateSignatureContext *ctx) = 0;
  virtual void exitQuantumGateSignature(qasm3Parser::QuantumGateSignatureContext *ctx) = 0;

  virtual void enterQuantumBlock(qasm3Parser::QuantumBlockContext *ctx) = 0;
  virtual void exitQuantumBlock(qasm3Parser::QuantumBlockContext *ctx) = 0;

  virtual void enterQuantumLoop(qasm3Parser::QuantumLoopContext *ctx) = 0;
  virtual void exitQuantumLoop(qasm3Parser::QuantumLoopContext *ctx) = 0;

  virtual void enterQuantumLoopBlock(qasm3Parser::QuantumLoopBlockContext *ctx) = 0;
  virtual void exitQuantumLoopBlock(qasm3Parser::QuantumLoopBlockContext *ctx) = 0;

  virtual void enterQuantumStatement(qasm3Parser::QuantumStatementContext *ctx) = 0;
  virtual void exitQuantumStatement(qasm3Parser::QuantumStatementContext *ctx) = 0;

  virtual void enterQuantumInstruction(qasm3Parser::QuantumInstructionContext *ctx) = 0;
  virtual void exitQuantumInstruction(qasm3Parser::QuantumInstructionContext *ctx) = 0;

  virtual void enterQuantumPhase(qasm3Parser::QuantumPhaseContext *ctx) = 0;
  virtual void exitQuantumPhase(qasm3Parser::QuantumPhaseContext *ctx) = 0;

  virtual void enterQuantumMeasurement(qasm3Parser::QuantumMeasurementContext *ctx) = 0;
  virtual void exitQuantumMeasurement(qasm3Parser::QuantumMeasurementContext *ctx) = 0;

  virtual void enterQuantumMeasurementAssignment(qasm3Parser::QuantumMeasurementAssignmentContext *ctx) = 0;
  virtual void exitQuantumMeasurementAssignment(qasm3Parser::QuantumMeasurementAssignmentContext *ctx) = 0;

  virtual void enterQuantumBarrier(qasm3Parser::QuantumBarrierContext *ctx) = 0;
  virtual void exitQuantumBarrier(qasm3Parser::QuantumBarrierContext *ctx) = 0;

  virtual void enterQuantumGateModifier(qasm3Parser::QuantumGateModifierContext *ctx) = 0;
  virtual void exitQuantumGateModifier(qasm3Parser::QuantumGateModifierContext *ctx) = 0;

  virtual void enterQuantumGateCall(qasm3Parser::QuantumGateCallContext *ctx) = 0;
  virtual void exitQuantumGateCall(qasm3Parser::QuantumGateCallContext *ctx) = 0;

  virtual void enterQuantumGateName(qasm3Parser::QuantumGateNameContext *ctx) = 0;
  virtual void exitQuantumGateName(qasm3Parser::QuantumGateNameContext *ctx) = 0;

  virtual void enterUnaryOperator(qasm3Parser::UnaryOperatorContext *ctx) = 0;
  virtual void exitUnaryOperator(qasm3Parser::UnaryOperatorContext *ctx) = 0;

  virtual void enterRelationalOperator(qasm3Parser::RelationalOperatorContext *ctx) = 0;
  virtual void exitRelationalOperator(qasm3Parser::RelationalOperatorContext *ctx) = 0;

  virtual void enterLogicalOperator(qasm3Parser::LogicalOperatorContext *ctx) = 0;
  virtual void exitLogicalOperator(qasm3Parser::LogicalOperatorContext *ctx) = 0;

  virtual void enterExpressionStatement(qasm3Parser::ExpressionStatementContext *ctx) = 0;
  virtual void exitExpressionStatement(qasm3Parser::ExpressionStatementContext *ctx) = 0;

  virtual void enterExpression(qasm3Parser::ExpressionContext *ctx) = 0;
  virtual void exitExpression(qasm3Parser::ExpressionContext *ctx) = 0;

  virtual void enterXOrExpression(qasm3Parser::XOrExpressionContext *ctx) = 0;
  virtual void exitXOrExpression(qasm3Parser::XOrExpressionContext *ctx) = 0;

  virtual void enterBitAndExpression(qasm3Parser::BitAndExpressionContext *ctx) = 0;
  virtual void exitBitAndExpression(qasm3Parser::BitAndExpressionContext *ctx) = 0;

  virtual void enterBitShiftExpression(qasm3Parser::BitShiftExpressionContext *ctx) = 0;
  virtual void exitBitShiftExpression(qasm3Parser::BitShiftExpressionContext *ctx) = 0;

  virtual void enterAdditiveExpression(qasm3Parser::AdditiveExpressionContext *ctx) = 0;
  virtual void exitAdditiveExpression(qasm3Parser::AdditiveExpressionContext *ctx) = 0;

  virtual void enterMultiplicativeExpression(qasm3Parser::MultiplicativeExpressionContext *ctx) = 0;
  virtual void exitMultiplicativeExpression(qasm3Parser::MultiplicativeExpressionContext *ctx) = 0;

  virtual void enterUnaryExpression(qasm3Parser::UnaryExpressionContext *ctx) = 0;
  virtual void exitUnaryExpression(qasm3Parser::UnaryExpressionContext *ctx) = 0;

  virtual void enterExpressionTerminator(qasm3Parser::ExpressionTerminatorContext *ctx) = 0;
  virtual void exitExpressionTerminator(qasm3Parser::ExpressionTerminatorContext *ctx) = 0;

  virtual void enterIncrementor(qasm3Parser::IncrementorContext *ctx) = 0;
  virtual void exitIncrementor(qasm3Parser::IncrementorContext *ctx) = 0;

  virtual void enterBuiltInCall(qasm3Parser::BuiltInCallContext *ctx) = 0;
  virtual void exitBuiltInCall(qasm3Parser::BuiltInCallContext *ctx) = 0;

  virtual void enterBuiltInMath(qasm3Parser::BuiltInMathContext *ctx) = 0;
  virtual void exitBuiltInMath(qasm3Parser::BuiltInMathContext *ctx) = 0;

  virtual void enterCastOperator(qasm3Parser::CastOperatorContext *ctx) = 0;
  virtual void exitCastOperator(qasm3Parser::CastOperatorContext *ctx) = 0;

  virtual void enterExpressionList(qasm3Parser::ExpressionListContext *ctx) = 0;
  virtual void exitExpressionList(qasm3Parser::ExpressionListContext *ctx) = 0;

  virtual void enterBooleanExpression(qasm3Parser::BooleanExpressionContext *ctx) = 0;
  virtual void exitBooleanExpression(qasm3Parser::BooleanExpressionContext *ctx) = 0;

  virtual void enterComparsionExpression(qasm3Parser::ComparsionExpressionContext *ctx) = 0;
  virtual void exitComparsionExpression(qasm3Parser::ComparsionExpressionContext *ctx) = 0;

  virtual void enterEqualsExpression(qasm3Parser::EqualsExpressionContext *ctx) = 0;
  virtual void exitEqualsExpression(qasm3Parser::EqualsExpressionContext *ctx) = 0;

  virtual void enterAssignmentOperator(qasm3Parser::AssignmentOperatorContext *ctx) = 0;
  virtual void exitAssignmentOperator(qasm3Parser::AssignmentOperatorContext *ctx) = 0;

  virtual void enterEqualsAssignmentList(qasm3Parser::EqualsAssignmentListContext *ctx) = 0;
  virtual void exitEqualsAssignmentList(qasm3Parser::EqualsAssignmentListContext *ctx) = 0;

  virtual void enterMembershipTest(qasm3Parser::MembershipTestContext *ctx) = 0;
  virtual void exitMembershipTest(qasm3Parser::MembershipTestContext *ctx) = 0;

  virtual void enterSetDeclaration(qasm3Parser::SetDeclarationContext *ctx) = 0;
  virtual void exitSetDeclaration(qasm3Parser::SetDeclarationContext *ctx) = 0;

  virtual void enterProgramBlock(qasm3Parser::ProgramBlockContext *ctx) = 0;
  virtual void exitProgramBlock(qasm3Parser::ProgramBlockContext *ctx) = 0;

  virtual void enterBranchingStatement(qasm3Parser::BranchingStatementContext *ctx) = 0;
  virtual void exitBranchingStatement(qasm3Parser::BranchingStatementContext *ctx) = 0;

  virtual void enterLoopSignature(qasm3Parser::LoopSignatureContext *ctx) = 0;
  virtual void exitLoopSignature(qasm3Parser::LoopSignatureContext *ctx) = 0;

  virtual void enterLoopStatement(qasm3Parser::LoopStatementContext *ctx) = 0;
  virtual void exitLoopStatement(qasm3Parser::LoopStatementContext *ctx) = 0;

  virtual void enterCLikeLoopStatement(qasm3Parser::CLikeLoopStatementContext *ctx) = 0;
  virtual void exitCLikeLoopStatement(qasm3Parser::CLikeLoopStatementContext *ctx) = 0;

  virtual void enterControlDirectiveStatement(qasm3Parser::ControlDirectiveStatementContext *ctx) = 0;
  virtual void exitControlDirectiveStatement(qasm3Parser::ControlDirectiveStatementContext *ctx) = 0;

  virtual void enterControlDirective(qasm3Parser::ControlDirectiveContext *ctx) = 0;
  virtual void exitControlDirective(qasm3Parser::ControlDirectiveContext *ctx) = 0;

  virtual void enterKernelDeclaration(qasm3Parser::KernelDeclarationContext *ctx) = 0;
  virtual void exitKernelDeclaration(qasm3Parser::KernelDeclarationContext *ctx) = 0;

  virtual void enterKernelCall(qasm3Parser::KernelCallContext *ctx) = 0;
  virtual void exitKernelCall(qasm3Parser::KernelCallContext *ctx) = 0;

  virtual void enterSubroutineDefinition(qasm3Parser::SubroutineDefinitionContext *ctx) = 0;
  virtual void exitSubroutineDefinition(qasm3Parser::SubroutineDefinitionContext *ctx) = 0;

  virtual void enterReturnStatement(qasm3Parser::ReturnStatementContext *ctx) = 0;
  virtual void exitReturnStatement(qasm3Parser::ReturnStatementContext *ctx) = 0;

  virtual void enterSubroutineBlock(qasm3Parser::SubroutineBlockContext *ctx) = 0;
  virtual void exitSubroutineBlock(qasm3Parser::SubroutineBlockContext *ctx) = 0;

  virtual void enterSubroutineCall(qasm3Parser::SubroutineCallContext *ctx) = 0;
  virtual void exitSubroutineCall(qasm3Parser::SubroutineCallContext *ctx) = 0;

  virtual void enterPragma(qasm3Parser::PragmaContext *ctx) = 0;
  virtual void exitPragma(qasm3Parser::PragmaContext *ctx) = 0;

  virtual void enterTimingType(qasm3Parser::TimingTypeContext *ctx) = 0;
  virtual void exitTimingType(qasm3Parser::TimingTypeContext *ctx) = 0;

  virtual void enterTimingBox(qasm3Parser::TimingBoxContext *ctx) = 0;
  virtual void exitTimingBox(qasm3Parser::TimingBoxContext *ctx) = 0;

  virtual void enterTimingTerminator(qasm3Parser::TimingTerminatorContext *ctx) = 0;
  virtual void exitTimingTerminator(qasm3Parser::TimingTerminatorContext *ctx) = 0;

  virtual void enterTimingIdentifier(qasm3Parser::TimingIdentifierContext *ctx) = 0;
  virtual void exitTimingIdentifier(qasm3Parser::TimingIdentifierContext *ctx) = 0;

  virtual void enterTimingInstructionName(qasm3Parser::TimingInstructionNameContext *ctx) = 0;
  virtual void exitTimingInstructionName(qasm3Parser::TimingInstructionNameContext *ctx) = 0;

  virtual void enterTimingInstruction(qasm3Parser::TimingInstructionContext *ctx) = 0;
  virtual void exitTimingInstruction(qasm3Parser::TimingInstructionContext *ctx) = 0;

  virtual void enterTimingStatement(qasm3Parser::TimingStatementContext *ctx) = 0;
  virtual void exitTimingStatement(qasm3Parser::TimingStatementContext *ctx) = 0;

  virtual void enterCalibration(qasm3Parser::CalibrationContext *ctx) = 0;
  virtual void exitCalibration(qasm3Parser::CalibrationContext *ctx) = 0;

  virtual void enterCalibrationGrammarDeclaration(qasm3Parser::CalibrationGrammarDeclarationContext *ctx) = 0;
  virtual void exitCalibrationGrammarDeclaration(qasm3Parser::CalibrationGrammarDeclarationContext *ctx) = 0;

  virtual void enterCalibrationDefinition(qasm3Parser::CalibrationDefinitionContext *ctx) = 0;
  virtual void exitCalibrationDefinition(qasm3Parser::CalibrationDefinitionContext *ctx) = 0;

  virtual void enterCalibrationGrammar(qasm3Parser::CalibrationGrammarContext *ctx) = 0;
  virtual void exitCalibrationGrammar(qasm3Parser::CalibrationGrammarContext *ctx) = 0;

  virtual void enterCalibrationArgumentList(qasm3Parser::CalibrationArgumentListContext *ctx) = 0;
  virtual void exitCalibrationArgumentList(qasm3Parser::CalibrationArgumentListContext *ctx) = 0;


};

}  // namespace qasm3

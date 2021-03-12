
// Generated from qasm3.g4 by ANTLR 4.9.2

#pragma once


#include "antlr4-runtime.h"
#include "qasm3Listener.h"


namespace qasm3 {

/**
 * This class provides an empty implementation of qasm3Listener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  qasm3BaseListener : public qasm3Listener {
public:

  virtual void enterProgram(qasm3Parser::ProgramContext * /*ctx*/) override { }
  virtual void exitProgram(qasm3Parser::ProgramContext * /*ctx*/) override { }

  virtual void enterHeader(qasm3Parser::HeaderContext * /*ctx*/) override { }
  virtual void exitHeader(qasm3Parser::HeaderContext * /*ctx*/) override { }

  virtual void enterVersion(qasm3Parser::VersionContext * /*ctx*/) override { }
  virtual void exitVersion(qasm3Parser::VersionContext * /*ctx*/) override { }

  virtual void enterInclude(qasm3Parser::IncludeContext * /*ctx*/) override { }
  virtual void exitInclude(qasm3Parser::IncludeContext * /*ctx*/) override { }

  virtual void enterGlobalStatement(qasm3Parser::GlobalStatementContext * /*ctx*/) override { }
  virtual void exitGlobalStatement(qasm3Parser::GlobalStatementContext * /*ctx*/) override { }

  virtual void enterStatement(qasm3Parser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(qasm3Parser::StatementContext * /*ctx*/) override { }

  virtual void enterQuantumDeclarationStatement(qasm3Parser::QuantumDeclarationStatementContext * /*ctx*/) override { }
  virtual void exitQuantumDeclarationStatement(qasm3Parser::QuantumDeclarationStatementContext * /*ctx*/) override { }

  virtual void enterClassicalDeclarationStatement(qasm3Parser::ClassicalDeclarationStatementContext * /*ctx*/) override { }
  virtual void exitClassicalDeclarationStatement(qasm3Parser::ClassicalDeclarationStatementContext * /*ctx*/) override { }

  virtual void enterClassicalAssignment(qasm3Parser::ClassicalAssignmentContext * /*ctx*/) override { }
  virtual void exitClassicalAssignment(qasm3Parser::ClassicalAssignmentContext * /*ctx*/) override { }

  virtual void enterAssignmentStatement(qasm3Parser::AssignmentStatementContext * /*ctx*/) override { }
  virtual void exitAssignmentStatement(qasm3Parser::AssignmentStatementContext * /*ctx*/) override { }

  virtual void enterReturnSignature(qasm3Parser::ReturnSignatureContext * /*ctx*/) override { }
  virtual void exitReturnSignature(qasm3Parser::ReturnSignatureContext * /*ctx*/) override { }

  virtual void enterDesignator(qasm3Parser::DesignatorContext * /*ctx*/) override { }
  virtual void exitDesignator(qasm3Parser::DesignatorContext * /*ctx*/) override { }

  virtual void enterDoubleDesignator(qasm3Parser::DoubleDesignatorContext * /*ctx*/) override { }
  virtual void exitDoubleDesignator(qasm3Parser::DoubleDesignatorContext * /*ctx*/) override { }

  virtual void enterIdentifierList(qasm3Parser::IdentifierListContext * /*ctx*/) override { }
  virtual void exitIdentifierList(qasm3Parser::IdentifierListContext * /*ctx*/) override { }

  virtual void enterAssociation(qasm3Parser::AssociationContext * /*ctx*/) override { }
  virtual void exitAssociation(qasm3Parser::AssociationContext * /*ctx*/) override { }

  virtual void enterQuantumType(qasm3Parser::QuantumTypeContext * /*ctx*/) override { }
  virtual void exitQuantumType(qasm3Parser::QuantumTypeContext * /*ctx*/) override { }

  virtual void enterQuantumDeclaration(qasm3Parser::QuantumDeclarationContext * /*ctx*/) override { }
  virtual void exitQuantumDeclaration(qasm3Parser::QuantumDeclarationContext * /*ctx*/) override { }

  virtual void enterQuantumArgument(qasm3Parser::QuantumArgumentContext * /*ctx*/) override { }
  virtual void exitQuantumArgument(qasm3Parser::QuantumArgumentContext * /*ctx*/) override { }

  virtual void enterQuantumArgumentList(qasm3Parser::QuantumArgumentListContext * /*ctx*/) override { }
  virtual void exitQuantumArgumentList(qasm3Parser::QuantumArgumentListContext * /*ctx*/) override { }

  virtual void enterBitType(qasm3Parser::BitTypeContext * /*ctx*/) override { }
  virtual void exitBitType(qasm3Parser::BitTypeContext * /*ctx*/) override { }

  virtual void enterSingleDesignatorType(qasm3Parser::SingleDesignatorTypeContext * /*ctx*/) override { }
  virtual void exitSingleDesignatorType(qasm3Parser::SingleDesignatorTypeContext * /*ctx*/) override { }

  virtual void enterDoubleDesignatorType(qasm3Parser::DoubleDesignatorTypeContext * /*ctx*/) override { }
  virtual void exitDoubleDesignatorType(qasm3Parser::DoubleDesignatorTypeContext * /*ctx*/) override { }

  virtual void enterNoDesignatorType(qasm3Parser::NoDesignatorTypeContext * /*ctx*/) override { }
  virtual void exitNoDesignatorType(qasm3Parser::NoDesignatorTypeContext * /*ctx*/) override { }

  virtual void enterClassicalType(qasm3Parser::ClassicalTypeContext * /*ctx*/) override { }
  virtual void exitClassicalType(qasm3Parser::ClassicalTypeContext * /*ctx*/) override { }

  virtual void enterConstantDeclaration(qasm3Parser::ConstantDeclarationContext * /*ctx*/) override { }
  virtual void exitConstantDeclaration(qasm3Parser::ConstantDeclarationContext * /*ctx*/) override { }

  virtual void enterSingleDesignatorDeclaration(qasm3Parser::SingleDesignatorDeclarationContext * /*ctx*/) override { }
  virtual void exitSingleDesignatorDeclaration(qasm3Parser::SingleDesignatorDeclarationContext * /*ctx*/) override { }

  virtual void enterDoubleDesignatorDeclaration(qasm3Parser::DoubleDesignatorDeclarationContext * /*ctx*/) override { }
  virtual void exitDoubleDesignatorDeclaration(qasm3Parser::DoubleDesignatorDeclarationContext * /*ctx*/) override { }

  virtual void enterNoDesignatorDeclaration(qasm3Parser::NoDesignatorDeclarationContext * /*ctx*/) override { }
  virtual void exitNoDesignatorDeclaration(qasm3Parser::NoDesignatorDeclarationContext * /*ctx*/) override { }

  virtual void enterBitDeclaration(qasm3Parser::BitDeclarationContext * /*ctx*/) override { }
  virtual void exitBitDeclaration(qasm3Parser::BitDeclarationContext * /*ctx*/) override { }

  virtual void enterClassicalDeclaration(qasm3Parser::ClassicalDeclarationContext * /*ctx*/) override { }
  virtual void exitClassicalDeclaration(qasm3Parser::ClassicalDeclarationContext * /*ctx*/) override { }

  virtual void enterClassicalTypeList(qasm3Parser::ClassicalTypeListContext * /*ctx*/) override { }
  virtual void exitClassicalTypeList(qasm3Parser::ClassicalTypeListContext * /*ctx*/) override { }

  virtual void enterClassicalArgument(qasm3Parser::ClassicalArgumentContext * /*ctx*/) override { }
  virtual void exitClassicalArgument(qasm3Parser::ClassicalArgumentContext * /*ctx*/) override { }

  virtual void enterClassicalArgumentList(qasm3Parser::ClassicalArgumentListContext * /*ctx*/) override { }
  virtual void exitClassicalArgumentList(qasm3Parser::ClassicalArgumentListContext * /*ctx*/) override { }

  virtual void enterAliasStatement(qasm3Parser::AliasStatementContext * /*ctx*/) override { }
  virtual void exitAliasStatement(qasm3Parser::AliasStatementContext * /*ctx*/) override { }

  virtual void enterIndexIdentifier(qasm3Parser::IndexIdentifierContext * /*ctx*/) override { }
  virtual void exitIndexIdentifier(qasm3Parser::IndexIdentifierContext * /*ctx*/) override { }

  virtual void enterIndexIdentifierList(qasm3Parser::IndexIdentifierListContext * /*ctx*/) override { }
  virtual void exitIndexIdentifierList(qasm3Parser::IndexIdentifierListContext * /*ctx*/) override { }

  virtual void enterIndexEqualsAssignmentList(qasm3Parser::IndexEqualsAssignmentListContext * /*ctx*/) override { }
  virtual void exitIndexEqualsAssignmentList(qasm3Parser::IndexEqualsAssignmentListContext * /*ctx*/) override { }

  virtual void enterRangeDefinition(qasm3Parser::RangeDefinitionContext * /*ctx*/) override { }
  virtual void exitRangeDefinition(qasm3Parser::RangeDefinitionContext * /*ctx*/) override { }

  virtual void enterQuantumGateDefinition(qasm3Parser::QuantumGateDefinitionContext * /*ctx*/) override { }
  virtual void exitQuantumGateDefinition(qasm3Parser::QuantumGateDefinitionContext * /*ctx*/) override { }

  virtual void enterQuantumGateSignature(qasm3Parser::QuantumGateSignatureContext * /*ctx*/) override { }
  virtual void exitQuantumGateSignature(qasm3Parser::QuantumGateSignatureContext * /*ctx*/) override { }

  virtual void enterQuantumBlock(qasm3Parser::QuantumBlockContext * /*ctx*/) override { }
  virtual void exitQuantumBlock(qasm3Parser::QuantumBlockContext * /*ctx*/) override { }

  virtual void enterQuantumLoop(qasm3Parser::QuantumLoopContext * /*ctx*/) override { }
  virtual void exitQuantumLoop(qasm3Parser::QuantumLoopContext * /*ctx*/) override { }

  virtual void enterQuantumLoopBlock(qasm3Parser::QuantumLoopBlockContext * /*ctx*/) override { }
  virtual void exitQuantumLoopBlock(qasm3Parser::QuantumLoopBlockContext * /*ctx*/) override { }

  virtual void enterQuantumStatement(qasm3Parser::QuantumStatementContext * /*ctx*/) override { }
  virtual void exitQuantumStatement(qasm3Parser::QuantumStatementContext * /*ctx*/) override { }

  virtual void enterQuantumInstruction(qasm3Parser::QuantumInstructionContext * /*ctx*/) override { }
  virtual void exitQuantumInstruction(qasm3Parser::QuantumInstructionContext * /*ctx*/) override { }

  virtual void enterQuantumPhase(qasm3Parser::QuantumPhaseContext * /*ctx*/) override { }
  virtual void exitQuantumPhase(qasm3Parser::QuantumPhaseContext * /*ctx*/) override { }

  virtual void enterQuantumMeasurement(qasm3Parser::QuantumMeasurementContext * /*ctx*/) override { }
  virtual void exitQuantumMeasurement(qasm3Parser::QuantumMeasurementContext * /*ctx*/) override { }

  virtual void enterQuantumMeasurementAssignment(qasm3Parser::QuantumMeasurementAssignmentContext * /*ctx*/) override { }
  virtual void exitQuantumMeasurementAssignment(qasm3Parser::QuantumMeasurementAssignmentContext * /*ctx*/) override { }

  virtual void enterQuantumBarrier(qasm3Parser::QuantumBarrierContext * /*ctx*/) override { }
  virtual void exitQuantumBarrier(qasm3Parser::QuantumBarrierContext * /*ctx*/) override { }

  virtual void enterQuantumGateModifier(qasm3Parser::QuantumGateModifierContext * /*ctx*/) override { }
  virtual void exitQuantumGateModifier(qasm3Parser::QuantumGateModifierContext * /*ctx*/) override { }

  virtual void enterQuantumGateCall(qasm3Parser::QuantumGateCallContext * /*ctx*/) override { }
  virtual void exitQuantumGateCall(qasm3Parser::QuantumGateCallContext * /*ctx*/) override { }

  virtual void enterQuantumGateName(qasm3Parser::QuantumGateNameContext * /*ctx*/) override { }
  virtual void exitQuantumGateName(qasm3Parser::QuantumGateNameContext * /*ctx*/) override { }

  virtual void enterUnaryOperator(qasm3Parser::UnaryOperatorContext * /*ctx*/) override { }
  virtual void exitUnaryOperator(qasm3Parser::UnaryOperatorContext * /*ctx*/) override { }

  virtual void enterRelationalOperator(qasm3Parser::RelationalOperatorContext * /*ctx*/) override { }
  virtual void exitRelationalOperator(qasm3Parser::RelationalOperatorContext * /*ctx*/) override { }

  virtual void enterLogicalOperator(qasm3Parser::LogicalOperatorContext * /*ctx*/) override { }
  virtual void exitLogicalOperator(qasm3Parser::LogicalOperatorContext * /*ctx*/) override { }

  virtual void enterExpressionStatement(qasm3Parser::ExpressionStatementContext * /*ctx*/) override { }
  virtual void exitExpressionStatement(qasm3Parser::ExpressionStatementContext * /*ctx*/) override { }

  virtual void enterExpression(qasm3Parser::ExpressionContext * /*ctx*/) override { }
  virtual void exitExpression(qasm3Parser::ExpressionContext * /*ctx*/) override { }

  virtual void enterXOrExpression(qasm3Parser::XOrExpressionContext * /*ctx*/) override { }
  virtual void exitXOrExpression(qasm3Parser::XOrExpressionContext * /*ctx*/) override { }

  virtual void enterBitAndExpression(qasm3Parser::BitAndExpressionContext * /*ctx*/) override { }
  virtual void exitBitAndExpression(qasm3Parser::BitAndExpressionContext * /*ctx*/) override { }

  virtual void enterBitShiftExpression(qasm3Parser::BitShiftExpressionContext * /*ctx*/) override { }
  virtual void exitBitShiftExpression(qasm3Parser::BitShiftExpressionContext * /*ctx*/) override { }

  virtual void enterAdditiveExpression(qasm3Parser::AdditiveExpressionContext * /*ctx*/) override { }
  virtual void exitAdditiveExpression(qasm3Parser::AdditiveExpressionContext * /*ctx*/) override { }

  virtual void enterMultiplicativeExpression(qasm3Parser::MultiplicativeExpressionContext * /*ctx*/) override { }
  virtual void exitMultiplicativeExpression(qasm3Parser::MultiplicativeExpressionContext * /*ctx*/) override { }

  virtual void enterUnaryExpression(qasm3Parser::UnaryExpressionContext * /*ctx*/) override { }
  virtual void exitUnaryExpression(qasm3Parser::UnaryExpressionContext * /*ctx*/) override { }

  virtual void enterExpressionTerminator(qasm3Parser::ExpressionTerminatorContext * /*ctx*/) override { }
  virtual void exitExpressionTerminator(qasm3Parser::ExpressionTerminatorContext * /*ctx*/) override { }

  virtual void enterIncrementor(qasm3Parser::IncrementorContext * /*ctx*/) override { }
  virtual void exitIncrementor(qasm3Parser::IncrementorContext * /*ctx*/) override { }

  virtual void enterBuiltInCall(qasm3Parser::BuiltInCallContext * /*ctx*/) override { }
  virtual void exitBuiltInCall(qasm3Parser::BuiltInCallContext * /*ctx*/) override { }

  virtual void enterBuiltInMath(qasm3Parser::BuiltInMathContext * /*ctx*/) override { }
  virtual void exitBuiltInMath(qasm3Parser::BuiltInMathContext * /*ctx*/) override { }

  virtual void enterCastOperator(qasm3Parser::CastOperatorContext * /*ctx*/) override { }
  virtual void exitCastOperator(qasm3Parser::CastOperatorContext * /*ctx*/) override { }

  virtual void enterExpressionList(qasm3Parser::ExpressionListContext * /*ctx*/) override { }
  virtual void exitExpressionList(qasm3Parser::ExpressionListContext * /*ctx*/) override { }

  virtual void enterBooleanExpression(qasm3Parser::BooleanExpressionContext * /*ctx*/) override { }
  virtual void exitBooleanExpression(qasm3Parser::BooleanExpressionContext * /*ctx*/) override { }

  virtual void enterComparsionExpression(qasm3Parser::ComparsionExpressionContext * /*ctx*/) override { }
  virtual void exitComparsionExpression(qasm3Parser::ComparsionExpressionContext * /*ctx*/) override { }

  virtual void enterEqualsExpression(qasm3Parser::EqualsExpressionContext * /*ctx*/) override { }
  virtual void exitEqualsExpression(qasm3Parser::EqualsExpressionContext * /*ctx*/) override { }

  virtual void enterAssignmentOperator(qasm3Parser::AssignmentOperatorContext * /*ctx*/) override { }
  virtual void exitAssignmentOperator(qasm3Parser::AssignmentOperatorContext * /*ctx*/) override { }

  virtual void enterEqualsAssignmentList(qasm3Parser::EqualsAssignmentListContext * /*ctx*/) override { }
  virtual void exitEqualsAssignmentList(qasm3Parser::EqualsAssignmentListContext * /*ctx*/) override { }

  virtual void enterMembershipTest(qasm3Parser::MembershipTestContext * /*ctx*/) override { }
  virtual void exitMembershipTest(qasm3Parser::MembershipTestContext * /*ctx*/) override { }

  virtual void enterSetDeclaration(qasm3Parser::SetDeclarationContext * /*ctx*/) override { }
  virtual void exitSetDeclaration(qasm3Parser::SetDeclarationContext * /*ctx*/) override { }

  virtual void enterProgramBlock(qasm3Parser::ProgramBlockContext * /*ctx*/) override { }
  virtual void exitProgramBlock(qasm3Parser::ProgramBlockContext * /*ctx*/) override { }

  virtual void enterBranchingStatement(qasm3Parser::BranchingStatementContext * /*ctx*/) override { }
  virtual void exitBranchingStatement(qasm3Parser::BranchingStatementContext * /*ctx*/) override { }

  virtual void enterLoopSignature(qasm3Parser::LoopSignatureContext * /*ctx*/) override { }
  virtual void exitLoopSignature(qasm3Parser::LoopSignatureContext * /*ctx*/) override { }

  virtual void enterLoopStatement(qasm3Parser::LoopStatementContext * /*ctx*/) override { }
  virtual void exitLoopStatement(qasm3Parser::LoopStatementContext * /*ctx*/) override { }

  virtual void enterControlDirectiveStatement(qasm3Parser::ControlDirectiveStatementContext * /*ctx*/) override { }
  virtual void exitControlDirectiveStatement(qasm3Parser::ControlDirectiveStatementContext * /*ctx*/) override { }

  virtual void enterControlDirective(qasm3Parser::ControlDirectiveContext * /*ctx*/) override { }
  virtual void exitControlDirective(qasm3Parser::ControlDirectiveContext * /*ctx*/) override { }

  virtual void enterKernelDeclaration(qasm3Parser::KernelDeclarationContext * /*ctx*/) override { }
  virtual void exitKernelDeclaration(qasm3Parser::KernelDeclarationContext * /*ctx*/) override { }

  virtual void enterKernelCall(qasm3Parser::KernelCallContext * /*ctx*/) override { }
  virtual void exitKernelCall(qasm3Parser::KernelCallContext * /*ctx*/) override { }

  virtual void enterSubroutineDefinition(qasm3Parser::SubroutineDefinitionContext * /*ctx*/) override { }
  virtual void exitSubroutineDefinition(qasm3Parser::SubroutineDefinitionContext * /*ctx*/) override { }

  virtual void enterReturnStatement(qasm3Parser::ReturnStatementContext * /*ctx*/) override { }
  virtual void exitReturnStatement(qasm3Parser::ReturnStatementContext * /*ctx*/) override { }

  virtual void enterSubroutineBlock(qasm3Parser::SubroutineBlockContext * /*ctx*/) override { }
  virtual void exitSubroutineBlock(qasm3Parser::SubroutineBlockContext * /*ctx*/) override { }

  virtual void enterSubroutineCall(qasm3Parser::SubroutineCallContext * /*ctx*/) override { }
  virtual void exitSubroutineCall(qasm3Parser::SubroutineCallContext * /*ctx*/) override { }

  virtual void enterPragma(qasm3Parser::PragmaContext * /*ctx*/) override { }
  virtual void exitPragma(qasm3Parser::PragmaContext * /*ctx*/) override { }

  virtual void enterTimingType(qasm3Parser::TimingTypeContext * /*ctx*/) override { }
  virtual void exitTimingType(qasm3Parser::TimingTypeContext * /*ctx*/) override { }

  virtual void enterTimingBox(qasm3Parser::TimingBoxContext * /*ctx*/) override { }
  virtual void exitTimingBox(qasm3Parser::TimingBoxContext * /*ctx*/) override { }

  virtual void enterTimingTerminator(qasm3Parser::TimingTerminatorContext * /*ctx*/) override { }
  virtual void exitTimingTerminator(qasm3Parser::TimingTerminatorContext * /*ctx*/) override { }

  virtual void enterTimingIdentifier(qasm3Parser::TimingIdentifierContext * /*ctx*/) override { }
  virtual void exitTimingIdentifier(qasm3Parser::TimingIdentifierContext * /*ctx*/) override { }

  virtual void enterTimingInstructionName(qasm3Parser::TimingInstructionNameContext * /*ctx*/) override { }
  virtual void exitTimingInstructionName(qasm3Parser::TimingInstructionNameContext * /*ctx*/) override { }

  virtual void enterTimingInstruction(qasm3Parser::TimingInstructionContext * /*ctx*/) override { }
  virtual void exitTimingInstruction(qasm3Parser::TimingInstructionContext * /*ctx*/) override { }

  virtual void enterTimingStatement(qasm3Parser::TimingStatementContext * /*ctx*/) override { }
  virtual void exitTimingStatement(qasm3Parser::TimingStatementContext * /*ctx*/) override { }

  virtual void enterCalibration(qasm3Parser::CalibrationContext * /*ctx*/) override { }
  virtual void exitCalibration(qasm3Parser::CalibrationContext * /*ctx*/) override { }

  virtual void enterCalibrationGrammarDeclaration(qasm3Parser::CalibrationGrammarDeclarationContext * /*ctx*/) override { }
  virtual void exitCalibrationGrammarDeclaration(qasm3Parser::CalibrationGrammarDeclarationContext * /*ctx*/) override { }

  virtual void enterCalibrationDefinition(qasm3Parser::CalibrationDefinitionContext * /*ctx*/) override { }
  virtual void exitCalibrationDefinition(qasm3Parser::CalibrationDefinitionContext * /*ctx*/) override { }

  virtual void enterCalibrationGrammar(qasm3Parser::CalibrationGrammarContext * /*ctx*/) override { }
  virtual void exitCalibrationGrammar(qasm3Parser::CalibrationGrammarContext * /*ctx*/) override { }

  virtual void enterCalibrationArgumentList(qasm3Parser::CalibrationArgumentListContext * /*ctx*/) override { }
  virtual void exitCalibrationArgumentList(qasm3Parser::CalibrationArgumentListContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

}  // namespace qasm3


// Generated from qasm3.g4 by ANTLR 4.9.1

#pragma once


#include "antlr4-runtime.h"
#include "qasm3Visitor.h"


namespace qasm3 {

/**
 * This class provides an empty implementation of qasm3Visitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  qasm3BaseVisitor : public qasm3Visitor {
public:

  virtual antlrcpp::Any visitProgram(qasm3Parser::ProgramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitHeader(qasm3Parser::HeaderContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVersion(qasm3Parser::VersionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitInclude(qasm3Parser::IncludeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitGlobalStatement(qasm3Parser::GlobalStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStatement(qasm3Parser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumDeclarationStatement(qasm3Parser::QuantumDeclarationStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassicalDeclarationStatement(qasm3Parser::ClassicalDeclarationStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassicalAssignment(qasm3Parser::ClassicalAssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssignmentStatement(qasm3Parser::AssignmentStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReturnSignature(qasm3Parser::ReturnSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDesignator(qasm3Parser::DesignatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDoubleDesignator(qasm3Parser::DoubleDesignatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdentifierList(qasm3Parser::IdentifierListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssociation(qasm3Parser::AssociationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumType(qasm3Parser::QuantumTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumDeclaration(qasm3Parser::QuantumDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumArgument(qasm3Parser::QuantumArgumentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumArgumentList(qasm3Parser::QuantumArgumentListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBitType(qasm3Parser::BitTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingleDesignatorType(qasm3Parser::SingleDesignatorTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDoubleDesignatorType(qasm3Parser::DoubleDesignatorTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNoDesignatorType(qasm3Parser::NoDesignatorTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassicalType(qasm3Parser::ClassicalTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitConstantDeclaration(qasm3Parser::ConstantDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingleDesignatorDeclaration(qasm3Parser::SingleDesignatorDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDoubleDesignatorDeclaration(qasm3Parser::DoubleDesignatorDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNoDesignatorDeclaration(qasm3Parser::NoDesignatorDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBitDeclaration(qasm3Parser::BitDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassicalDeclaration(qasm3Parser::ClassicalDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassicalTypeList(qasm3Parser::ClassicalTypeListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassicalArgument(qasm3Parser::ClassicalArgumentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitClassicalArgumentList(qasm3Parser::ClassicalArgumentListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAliasStatement(qasm3Parser::AliasStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIndexIdentifier(qasm3Parser::IndexIdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIndexIdentifierList(qasm3Parser::IndexIdentifierListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIndexEqualsAssignmentList(qasm3Parser::IndexEqualsAssignmentListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitRangeDefinition(qasm3Parser::RangeDefinitionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumGateDefinition(qasm3Parser::QuantumGateDefinitionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumGateSignature(qasm3Parser::QuantumGateSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumBlock(qasm3Parser::QuantumBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumLoop(qasm3Parser::QuantumLoopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumLoopBlock(qasm3Parser::QuantumLoopBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumStatement(qasm3Parser::QuantumStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumInstruction(qasm3Parser::QuantumInstructionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumPhase(qasm3Parser::QuantumPhaseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumMeasurement(qasm3Parser::QuantumMeasurementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumMeasurementAssignment(qasm3Parser::QuantumMeasurementAssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumBarrier(qasm3Parser::QuantumBarrierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumGateModifier(qasm3Parser::QuantumGateModifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumGateCall(qasm3Parser::QuantumGateCallContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantumGateName(qasm3Parser::QuantumGateNameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnaryOperator(qasm3Parser::UnaryOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBinaryOperator(qasm3Parser::BinaryOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpressionStatement(qasm3Parser::ExpressionStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression(qasm3Parser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpressionTerminator(qasm3Parser::ExpressionTerminatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpressionList(qasm3Parser::ExpressionListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBuiltInCall(qasm3Parser::BuiltInCallContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBuiltInMath(qasm3Parser::BuiltInMathContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCastOperator(qasm3Parser::CastOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIncrementor(qasm3Parser::IncrementorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitEqualsExpression(qasm3Parser::EqualsExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssignmentOperator(qasm3Parser::AssignmentOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitEqualsAssignmentList(qasm3Parser::EqualsAssignmentListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMembershipTest(qasm3Parser::MembershipTestContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSetDeclaration(qasm3Parser::SetDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitProgramBlock(qasm3Parser::ProgramBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBranchingStatement(qasm3Parser::BranchingStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLoopSignature(qasm3Parser::LoopSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLoopStatement(qasm3Parser::LoopStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitControlDirectiveStatement(qasm3Parser::ControlDirectiveStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitControlDirective(qasm3Parser::ControlDirectiveContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitKernelDeclaration(qasm3Parser::KernelDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitKernelCall(qasm3Parser::KernelCallContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSubroutineDefinition(qasm3Parser::SubroutineDefinitionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReturnStatement(qasm3Parser::ReturnStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSubroutineBlock(qasm3Parser::SubroutineBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSubroutineCall(qasm3Parser::SubroutineCallContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPragma(qasm3Parser::PragmaContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTimingType(qasm3Parser::TimingTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTimingBox(qasm3Parser::TimingBoxContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTimingTerminator(qasm3Parser::TimingTerminatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTimingIdentifier(qasm3Parser::TimingIdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTimingInstructionName(qasm3Parser::TimingInstructionNameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTimingInstruction(qasm3Parser::TimingInstructionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTimingStatement(qasm3Parser::TimingStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCalibration(qasm3Parser::CalibrationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCalibrationGrammarDeclaration(qasm3Parser::CalibrationGrammarDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCalibrationDefinition(qasm3Parser::CalibrationDefinitionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCalibrationGrammar(qasm3Parser::CalibrationGrammarContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCalibrationArgumentList(qasm3Parser::CalibrationArgumentListContext *ctx) override {
    return visitChildren(ctx);
  }


};

}  // namespace qasm3

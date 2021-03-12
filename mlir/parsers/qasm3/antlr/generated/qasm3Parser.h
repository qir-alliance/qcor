
// Generated from qasm3.g4 by ANTLR 4.9.2

#pragma once


#include "antlr4-runtime.h"


namespace qasm3 {


class  qasm3Parser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, T__24 = 25, T__25 = 26, 
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, T__30 = 31, T__31 = 32, 
    T__32 = 33, T__33 = 34, T__34 = 35, T__35 = 36, T__36 = 37, T__37 = 38, 
    T__38 = 39, T__39 = 40, T__40 = 41, T__41 = 42, T__42 = 43, T__43 = 44, 
    T__44 = 45, T__45 = 46, T__46 = 47, T__47 = 48, T__48 = 49, T__49 = 50, 
    T__50 = 51, T__51 = 52, T__52 = 53, T__53 = 54, T__54 = 55, T__55 = 56, 
    T__56 = 57, T__57 = 58, T__58 = 59, T__59 = 60, T__60 = 61, T__61 = 62, 
    T__62 = 63, T__63 = 64, T__64 = 65, T__65 = 66, T__66 = 67, T__67 = 68, 
    T__68 = 69, T__69 = 70, T__70 = 71, T__71 = 72, T__72 = 73, T__73 = 74, 
    T__74 = 75, T__75 = 76, T__76 = 77, T__77 = 78, T__78 = 79, T__79 = 80, 
    T__80 = 81, T__81 = 82, T__82 = 83, T__83 = 84, T__84 = 85, T__85 = 86, 
    T__86 = 87, T__87 = 88, T__88 = 89, T__89 = 90, T__90 = 91, LBRACKET = 92, 
    RBRACKET = 93, LBRACE = 94, RBRACE = 95, LPAREN = 96, RPAREN = 97, COLON = 98, 
    SEMICOLON = 99, DOT = 100, COMMA = 101, EQUALS = 102, ARROW = 103, MINUS = 104, 
    Constant = 105, Whitespace = 106, Newline = 107, Integer = 108, Identifier = 109, 
    RealNumber = 110, TimingLiteral = 111, StringLiteral = 112, LineComment = 113, 
    BlockComment = 114
  };

  enum {
    RuleProgram = 0, RuleHeader = 1, RuleVersion = 2, RuleInclude = 3, RuleGlobalStatement = 4, 
    RuleStatement = 5, RuleQuantumDeclarationStatement = 6, RuleClassicalDeclarationStatement = 7, 
    RuleClassicalAssignment = 8, RuleAssignmentStatement = 9, RuleReturnSignature = 10, 
    RuleDesignator = 11, RuleDoubleDesignator = 12, RuleIdentifierList = 13, 
    RuleAssociation = 14, RuleQuantumType = 15, RuleQuantumDeclaration = 16, 
    RuleQuantumArgument = 17, RuleQuantumArgumentList = 18, RuleBitType = 19, 
    RuleSingleDesignatorType = 20, RuleDoubleDesignatorType = 21, RuleNoDesignatorType = 22, 
    RuleClassicalType = 23, RuleConstantDeclaration = 24, RuleSingleDesignatorDeclaration = 25, 
    RuleDoubleDesignatorDeclaration = 26, RuleNoDesignatorDeclaration = 27, 
    RuleBitDeclaration = 28, RuleClassicalDeclaration = 29, RuleClassicalTypeList = 30, 
    RuleClassicalArgument = 31, RuleClassicalArgumentList = 32, RuleAliasStatement = 33, 
    RuleIndexIdentifier = 34, RuleIndexIdentifierList = 35, RuleIndexEqualsAssignmentList = 36, 
    RuleRangeDefinition = 37, RuleQuantumGateDefinition = 38, RuleQuantumGateSignature = 39, 
    RuleQuantumBlock = 40, RuleQuantumLoop = 41, RuleQuantumLoopBlock = 42, 
    RuleQuantumStatement = 43, RuleQuantumInstruction = 44, RuleQuantumPhase = 45, 
    RuleQuantumMeasurement = 46, RuleQuantumMeasurementAssignment = 47, 
    RuleQuantumBarrier = 48, RuleQuantumGateModifier = 49, RuleQuantumGateCall = 50, 
    RuleQuantumGateName = 51, RuleUnaryOperator = 52, RuleRelationalOperator = 53, 
    RuleLogicalOperator = 54, RuleExpressionStatement = 55, RuleExpression = 56, 
    RuleXOrExpression = 57, RuleBitAndExpression = 58, RuleBitShiftExpression = 59, 
    RuleAdditiveExpression = 60, RuleMultiplicativeExpression = 61, RuleUnaryExpression = 62, 
    RuleExpressionTerminator = 63, RuleIncrementor = 64, RuleBuiltInCall = 65, 
    RuleBuiltInMath = 66, RuleCastOperator = 67, RuleExpressionList = 68, 
    RuleBooleanExpression = 69, RuleComparsionExpression = 70, RuleEqualsExpression = 71, 
    RuleAssignmentOperator = 72, RuleEqualsAssignmentList = 73, RuleMembershipTest = 74, 
    RuleSetDeclaration = 75, RuleProgramBlock = 76, RuleBranchingStatement = 77, 
    RuleLoopSignature = 78, RuleLoopStatement = 79, RuleControlDirectiveStatement = 80, 
    RuleControlDirective = 81, RuleKernelDeclaration = 82, RuleKernelCall = 83, 
    RuleSubroutineDefinition = 84, RuleReturnStatement = 85, RuleSubroutineBlock = 86, 
    RuleSubroutineCall = 87, RulePragma = 88, RuleTimingType = 89, RuleTimingBox = 90, 
    RuleTimingTerminator = 91, RuleTimingIdentifier = 92, RuleTimingInstructionName = 93, 
    RuleTimingInstruction = 94, RuleTimingStatement = 95, RuleCalibration = 96, 
    RuleCalibrationGrammarDeclaration = 97, RuleCalibrationDefinition = 98, 
    RuleCalibrationGrammar = 99, RuleCalibrationArgumentList = 100
  };

  explicit qasm3Parser(antlr4::TokenStream *input);
  ~qasm3Parser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class ProgramContext;
  class HeaderContext;
  class VersionContext;
  class IncludeContext;
  class GlobalStatementContext;
  class StatementContext;
  class QuantumDeclarationStatementContext;
  class ClassicalDeclarationStatementContext;
  class ClassicalAssignmentContext;
  class AssignmentStatementContext;
  class ReturnSignatureContext;
  class DesignatorContext;
  class DoubleDesignatorContext;
  class IdentifierListContext;
  class AssociationContext;
  class QuantumTypeContext;
  class QuantumDeclarationContext;
  class QuantumArgumentContext;
  class QuantumArgumentListContext;
  class BitTypeContext;
  class SingleDesignatorTypeContext;
  class DoubleDesignatorTypeContext;
  class NoDesignatorTypeContext;
  class ClassicalTypeContext;
  class ConstantDeclarationContext;
  class SingleDesignatorDeclarationContext;
  class DoubleDesignatorDeclarationContext;
  class NoDesignatorDeclarationContext;
  class BitDeclarationContext;
  class ClassicalDeclarationContext;
  class ClassicalTypeListContext;
  class ClassicalArgumentContext;
  class ClassicalArgumentListContext;
  class AliasStatementContext;
  class IndexIdentifierContext;
  class IndexIdentifierListContext;
  class IndexEqualsAssignmentListContext;
  class RangeDefinitionContext;
  class QuantumGateDefinitionContext;
  class QuantumGateSignatureContext;
  class QuantumBlockContext;
  class QuantumLoopContext;
  class QuantumLoopBlockContext;
  class QuantumStatementContext;
  class QuantumInstructionContext;
  class QuantumPhaseContext;
  class QuantumMeasurementContext;
  class QuantumMeasurementAssignmentContext;
  class QuantumBarrierContext;
  class QuantumGateModifierContext;
  class QuantumGateCallContext;
  class QuantumGateNameContext;
  class UnaryOperatorContext;
  class RelationalOperatorContext;
  class LogicalOperatorContext;
  class ExpressionStatementContext;
  class ExpressionContext;
  class XOrExpressionContext;
  class BitAndExpressionContext;
  class BitShiftExpressionContext;
  class AdditiveExpressionContext;
  class MultiplicativeExpressionContext;
  class UnaryExpressionContext;
  class ExpressionTerminatorContext;
  class IncrementorContext;
  class BuiltInCallContext;
  class BuiltInMathContext;
  class CastOperatorContext;
  class ExpressionListContext;
  class BooleanExpressionContext;
  class ComparsionExpressionContext;
  class EqualsExpressionContext;
  class AssignmentOperatorContext;
  class EqualsAssignmentListContext;
  class MembershipTestContext;
  class SetDeclarationContext;
  class ProgramBlockContext;
  class BranchingStatementContext;
  class LoopSignatureContext;
  class LoopStatementContext;
  class ControlDirectiveStatementContext;
  class ControlDirectiveContext;
  class KernelDeclarationContext;
  class KernelCallContext;
  class SubroutineDefinitionContext;
  class ReturnStatementContext;
  class SubroutineBlockContext;
  class SubroutineCallContext;
  class PragmaContext;
  class TimingTypeContext;
  class TimingBoxContext;
  class TimingTerminatorContext;
  class TimingIdentifierContext;
  class TimingInstructionNameContext;
  class TimingInstructionContext;
  class TimingStatementContext;
  class CalibrationContext;
  class CalibrationGrammarDeclarationContext;
  class CalibrationDefinitionContext;
  class CalibrationGrammarContext;
  class CalibrationArgumentListContext; 

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    HeaderContext *header();
    std::vector<GlobalStatementContext *> globalStatement();
    GlobalStatementContext* globalStatement(size_t i);
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgramContext* program();

  class  HeaderContext : public antlr4::ParserRuleContext {
  public:
    HeaderContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VersionContext *version();
    std::vector<IncludeContext *> include();
    IncludeContext* include(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  HeaderContext* header();

  class  VersionContext : public antlr4::ParserRuleContext {
  public:
    VersionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *SEMICOLON();
    antlr4::tree::TerminalNode *Integer();
    antlr4::tree::TerminalNode *RealNumber();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  VersionContext* version();

  class  IncludeContext : public antlr4::ParserRuleContext {
  public:
    IncludeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *StringLiteral();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IncludeContext* include();

  class  GlobalStatementContext : public antlr4::ParserRuleContext {
  public:
    GlobalStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SubroutineDefinitionContext *subroutineDefinition();
    KernelDeclarationContext *kernelDeclaration();
    QuantumGateDefinitionContext *quantumGateDefinition();
    CalibrationContext *calibration();
    QuantumDeclarationStatementContext *quantumDeclarationStatement();
    PragmaContext *pragma();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  GlobalStatementContext* globalStatement();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionStatementContext *expressionStatement();
    AssignmentStatementContext *assignmentStatement();
    ClassicalDeclarationStatementContext *classicalDeclarationStatement();
    BranchingStatementContext *branchingStatement();
    LoopStatementContext *loopStatement();
    ControlDirectiveStatementContext *controlDirectiveStatement();
    AliasStatementContext *aliasStatement();
    QuantumStatementContext *quantumStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StatementContext* statement();

  class  QuantumDeclarationStatementContext : public antlr4::ParserRuleContext {
  public:
    QuantumDeclarationStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumDeclarationContext *quantumDeclaration();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumDeclarationStatementContext* quantumDeclarationStatement();

  class  ClassicalDeclarationStatementContext : public antlr4::ParserRuleContext {
  public:
    ClassicalDeclarationStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *SEMICOLON();
    ClassicalDeclarationContext *classicalDeclaration();
    ConstantDeclarationContext *constantDeclaration();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ClassicalDeclarationStatementContext* classicalDeclarationStatement();

  class  ClassicalAssignmentContext : public antlr4::ParserRuleContext {
  public:
    ClassicalAssignmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IndexIdentifierContext *> indexIdentifier();
    IndexIdentifierContext* indexIdentifier(size_t i);
    AssignmentOperatorContext *assignmentOperator();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ClassicalAssignmentContext* classicalAssignment();

  class  AssignmentStatementContext : public antlr4::ParserRuleContext {
  public:
    AssignmentStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *SEMICOLON();
    ClassicalAssignmentContext *classicalAssignment();
    QuantumMeasurementAssignmentContext *quantumMeasurementAssignment();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AssignmentStatementContext* assignmentStatement();

  class  ReturnSignatureContext : public antlr4::ParserRuleContext {
  public:
    ReturnSignatureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ARROW();
    ClassicalTypeContext *classicalType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ReturnSignatureContext* returnSignature();

  class  DesignatorContext : public antlr4::ParserRuleContext {
  public:
    DesignatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACKET();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RBRACKET();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DesignatorContext* designator();

  class  DoubleDesignatorContext : public antlr4::ParserRuleContext {
  public:
    DoubleDesignatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACKET();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RBRACKET();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DoubleDesignatorContext* doubleDesignator();

  class  IdentifierListContext : public antlr4::ParserRuleContext {
  public:
    IdentifierListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> Identifier();
    antlr4::tree::TerminalNode* Identifier(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IdentifierListContext* identifierList();

  class  AssociationContext : public antlr4::ParserRuleContext {
  public:
    AssociationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *COLON();
    antlr4::tree::TerminalNode *Identifier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AssociationContext* association();

  class  QuantumTypeContext : public antlr4::ParserRuleContext {
  public:
    QuantumTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumTypeContext* quantumType();

  class  QuantumDeclarationContext : public antlr4::ParserRuleContext {
  public:
    QuantumDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumTypeContext *quantumType();
    IndexIdentifierListContext *indexIdentifierList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumDeclarationContext* quantumDeclaration();

  class  QuantumArgumentContext : public antlr4::ParserRuleContext {
  public:
    QuantumArgumentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumTypeContext *quantumType();
    AssociationContext *association();
    DesignatorContext *designator();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumArgumentContext* quantumArgument();

  class  QuantumArgumentListContext : public antlr4::ParserRuleContext {
  public:
    QuantumArgumentListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<QuantumArgumentContext *> quantumArgument();
    QuantumArgumentContext* quantumArgument(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumArgumentListContext* quantumArgumentList();

  class  BitTypeContext : public antlr4::ParserRuleContext {
  public:
    BitTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BitTypeContext* bitType();

  class  SingleDesignatorTypeContext : public antlr4::ParserRuleContext {
  public:
    SingleDesignatorTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SingleDesignatorTypeContext* singleDesignatorType();

  class  DoubleDesignatorTypeContext : public antlr4::ParserRuleContext {
  public:
    DoubleDesignatorTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DoubleDesignatorTypeContext* doubleDesignatorType();

  class  NoDesignatorTypeContext : public antlr4::ParserRuleContext {
  public:
    NoDesignatorTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TimingTypeContext *timingType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  NoDesignatorTypeContext* noDesignatorType();

  class  ClassicalTypeContext : public antlr4::ParserRuleContext {
  public:
    ClassicalTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SingleDesignatorTypeContext *singleDesignatorType();
    DesignatorContext *designator();
    DoubleDesignatorTypeContext *doubleDesignatorType();
    DoubleDesignatorContext *doubleDesignator();
    NoDesignatorTypeContext *noDesignatorType();
    BitTypeContext *bitType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ClassicalTypeContext* classicalType();

  class  ConstantDeclarationContext : public antlr4::ParserRuleContext {
  public:
    ConstantDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    EqualsAssignmentListContext *equalsAssignmentList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ConstantDeclarationContext* constantDeclaration();

  class  SingleDesignatorDeclarationContext : public antlr4::ParserRuleContext {
  public:
    SingleDesignatorDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SingleDesignatorTypeContext *singleDesignatorType();
    DesignatorContext *designator();
    IdentifierListContext *identifierList();
    EqualsAssignmentListContext *equalsAssignmentList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SingleDesignatorDeclarationContext* singleDesignatorDeclaration();

  class  DoubleDesignatorDeclarationContext : public antlr4::ParserRuleContext {
  public:
    DoubleDesignatorDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DoubleDesignatorTypeContext *doubleDesignatorType();
    DoubleDesignatorContext *doubleDesignator();
    IdentifierListContext *identifierList();
    EqualsAssignmentListContext *equalsAssignmentList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DoubleDesignatorDeclarationContext* doubleDesignatorDeclaration();

  class  NoDesignatorDeclarationContext : public antlr4::ParserRuleContext {
  public:
    NoDesignatorDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    NoDesignatorTypeContext *noDesignatorType();
    IdentifierListContext *identifierList();
    EqualsAssignmentListContext *equalsAssignmentList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  NoDesignatorDeclarationContext* noDesignatorDeclaration();

  class  BitDeclarationContext : public antlr4::ParserRuleContext {
  public:
    BitDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BitTypeContext *bitType();
    IndexIdentifierListContext *indexIdentifierList();
    IndexEqualsAssignmentListContext *indexEqualsAssignmentList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BitDeclarationContext* bitDeclaration();

  class  ClassicalDeclarationContext : public antlr4::ParserRuleContext {
  public:
    ClassicalDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SingleDesignatorDeclarationContext *singleDesignatorDeclaration();
    DoubleDesignatorDeclarationContext *doubleDesignatorDeclaration();
    NoDesignatorDeclarationContext *noDesignatorDeclaration();
    BitDeclarationContext *bitDeclaration();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ClassicalDeclarationContext* classicalDeclaration();

  class  ClassicalTypeListContext : public antlr4::ParserRuleContext {
  public:
    ClassicalTypeListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ClassicalTypeContext *> classicalType();
    ClassicalTypeContext* classicalType(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ClassicalTypeListContext* classicalTypeList();

  class  ClassicalArgumentContext : public antlr4::ParserRuleContext {
  public:
    ClassicalArgumentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ClassicalTypeContext *classicalType();
    AssociationContext *association();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ClassicalArgumentContext* classicalArgument();

  class  ClassicalArgumentListContext : public antlr4::ParserRuleContext {
  public:
    ClassicalArgumentListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ClassicalArgumentContext *> classicalArgument();
    ClassicalArgumentContext* classicalArgument(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ClassicalArgumentListContext* classicalArgumentList();

  class  AliasStatementContext : public antlr4::ParserRuleContext {
  public:
    AliasStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *EQUALS();
    IndexIdentifierContext *indexIdentifier();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AliasStatementContext* aliasStatement();

  class  IndexIdentifierContext : public antlr4::ParserRuleContext {
  public:
    IndexIdentifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    RangeDefinitionContext *rangeDefinition();
    antlr4::tree::TerminalNode *LBRACKET();
    ExpressionListContext *expressionList();
    antlr4::tree::TerminalNode *RBRACKET();
    std::vector<IndexIdentifierContext *> indexIdentifier();
    IndexIdentifierContext* indexIdentifier(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IndexIdentifierContext* indexIdentifier();
  IndexIdentifierContext* indexIdentifier(int precedence);
  class  IndexIdentifierListContext : public antlr4::ParserRuleContext {
  public:
    IndexIdentifierListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IndexIdentifierContext *> indexIdentifier();
    IndexIdentifierContext* indexIdentifier(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IndexIdentifierListContext* indexIdentifierList();

  class  IndexEqualsAssignmentListContext : public antlr4::ParserRuleContext {
  public:
    IndexEqualsAssignmentListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IndexIdentifierContext *> indexIdentifier();
    IndexIdentifierContext* indexIdentifier(size_t i);
    std::vector<EqualsExpressionContext *> equalsExpression();
    EqualsExpressionContext* equalsExpression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IndexEqualsAssignmentListContext* indexEqualsAssignmentList();

  class  RangeDefinitionContext : public antlr4::ParserRuleContext {
  public:
    RangeDefinitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACKET();
    std::vector<antlr4::tree::TerminalNode *> COLON();
    antlr4::tree::TerminalNode* COLON(size_t i);
    antlr4::tree::TerminalNode *RBRACKET();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RangeDefinitionContext* rangeDefinition();

  class  QuantumGateDefinitionContext : public antlr4::ParserRuleContext {
  public:
    QuantumGateDefinitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumGateSignatureContext *quantumGateSignature();
    QuantumBlockContext *quantumBlock();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumGateDefinitionContext* quantumGateDefinition();

  class  QuantumGateSignatureContext : public antlr4::ParserRuleContext {
  public:
    QuantumGateSignatureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IdentifierListContext *> identifierList();
    IdentifierListContext* identifierList(size_t i);
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumGateSignatureContext* quantumGateSignature();

  class  QuantumBlockContext : public antlr4::ParserRuleContext {
  public:
    QuantumBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACE();
    antlr4::tree::TerminalNode *RBRACE();
    std::vector<QuantumStatementContext *> quantumStatement();
    QuantumStatementContext* quantumStatement(size_t i);
    std::vector<QuantumLoopContext *> quantumLoop();
    QuantumLoopContext* quantumLoop(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumBlockContext* quantumBlock();

  class  QuantumLoopContext : public antlr4::ParserRuleContext {
  public:
    QuantumLoopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LoopSignatureContext *loopSignature();
    QuantumLoopBlockContext *quantumLoopBlock();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumLoopContext* quantumLoop();

  class  QuantumLoopBlockContext : public antlr4::ParserRuleContext {
  public:
    QuantumLoopBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<QuantumStatementContext *> quantumStatement();
    QuantumStatementContext* quantumStatement(size_t i);
    antlr4::tree::TerminalNode *LBRACE();
    antlr4::tree::TerminalNode *RBRACE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumLoopBlockContext* quantumLoopBlock();

  class  QuantumStatementContext : public antlr4::ParserRuleContext {
  public:
    QuantumStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumInstructionContext *quantumInstruction();
    antlr4::tree::TerminalNode *SEMICOLON();
    TimingStatementContext *timingStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumStatementContext* quantumStatement();

  class  QuantumInstructionContext : public antlr4::ParserRuleContext {
  public:
    QuantumInstructionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumGateCallContext *quantumGateCall();
    QuantumPhaseContext *quantumPhase();
    QuantumMeasurementContext *quantumMeasurement();
    QuantumBarrierContext *quantumBarrier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumInstructionContext* quantumInstruction();

  class  QuantumPhaseContext : public antlr4::ParserRuleContext {
  public:
    QuantumPhaseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumPhaseContext* quantumPhase();

  class  QuantumMeasurementContext : public antlr4::ParserRuleContext {
  public:
    QuantumMeasurementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IndexIdentifierListContext *indexIdentifierList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumMeasurementContext* quantumMeasurement();

  class  QuantumMeasurementAssignmentContext : public antlr4::ParserRuleContext {
  public:
    QuantumMeasurementAssignmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumMeasurementContext *quantumMeasurement();
    antlr4::tree::TerminalNode *ARROW();
    IndexIdentifierListContext *indexIdentifierList();
    antlr4::tree::TerminalNode *EQUALS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumMeasurementAssignmentContext* quantumMeasurementAssignment();

  class  QuantumBarrierContext : public antlr4::ParserRuleContext {
  public:
    QuantumBarrierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IndexIdentifierListContext *indexIdentifierList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumBarrierContext* quantumBarrier();

  class  QuantumGateModifierContext : public antlr4::ParserRuleContext {
  public:
    QuantumGateModifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACKET();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RBRACKET();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumGateModifierContext* quantumGateModifier();

  class  QuantumGateCallContext : public antlr4::ParserRuleContext {
  public:
    QuantumGateCallContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumGateNameContext *quantumGateName();
    IndexIdentifierListContext *indexIdentifierList();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ExpressionListContext *expressionList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumGateCallContext* quantumGateCall();

  class  QuantumGateNameContext : public antlr4::ParserRuleContext {
  public:
    QuantumGateNameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    QuantumGateModifierContext *quantumGateModifier();
    QuantumGateNameContext *quantumGateName();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QuantumGateNameContext* quantumGateName();

  class  UnaryOperatorContext : public antlr4::ParserRuleContext {
  public:
    UnaryOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  UnaryOperatorContext* unaryOperator();

  class  RelationalOperatorContext : public antlr4::ParserRuleContext {
  public:
    RelationalOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RelationalOperatorContext* relationalOperator();

  class  LogicalOperatorContext : public antlr4::ParserRuleContext {
  public:
    LogicalOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LogicalOperatorContext* logicalOperator();

  class  ExpressionStatementContext : public antlr4::ParserRuleContext {
  public:
    ExpressionStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionStatementContext* expressionStatement();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionTerminatorContext *expressionTerminator();
    UnaryExpressionContext *unaryExpression();
    XOrExpressionContext *xOrExpression();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();
  ExpressionContext* expression(int precedence);
  class  XOrExpressionContext : public antlr4::ParserRuleContext {
  public:
    XOrExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BitAndExpressionContext *bitAndExpression();
    XOrExpressionContext *xOrExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  XOrExpressionContext* xOrExpression();
  XOrExpressionContext* xOrExpression(int precedence);
  class  BitAndExpressionContext : public antlr4::ParserRuleContext {
  public:
    BitAndExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BitShiftExpressionContext *bitShiftExpression();
    BitAndExpressionContext *bitAndExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BitAndExpressionContext* bitAndExpression();
  BitAndExpressionContext* bitAndExpression(int precedence);
  class  BitShiftExpressionContext : public antlr4::ParserRuleContext {
  public:
    BitShiftExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    AdditiveExpressionContext *additiveExpression();
    BitShiftExpressionContext *bitShiftExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BitShiftExpressionContext* bitShiftExpression();
  BitShiftExpressionContext* bitShiftExpression(int precedence);
  class  AdditiveExpressionContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *binary_op = nullptr;
    AdditiveExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MultiplicativeExpressionContext *multiplicativeExpression();
    AdditiveExpressionContext *additiveExpression();
    antlr4::tree::TerminalNode *MINUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AdditiveExpressionContext* additiveExpression();
  AdditiveExpressionContext* additiveExpression(int precedence);
  class  MultiplicativeExpressionContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *binary_op = nullptr;
    MultiplicativeExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionTerminatorContext *expressionTerminator();
    UnaryExpressionContext *unaryExpression();
    MultiplicativeExpressionContext *multiplicativeExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  MultiplicativeExpressionContext* multiplicativeExpression();
  MultiplicativeExpressionContext* multiplicativeExpression(int precedence);
  class  UnaryExpressionContext : public antlr4::ParserRuleContext {
  public:
    UnaryExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UnaryOperatorContext *unaryOperator();
    ExpressionTerminatorContext *expressionTerminator();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  UnaryExpressionContext* unaryExpression();

  class  ExpressionTerminatorContext : public antlr4::ParserRuleContext {
  public:
    ExpressionTerminatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Constant();
    antlr4::tree::TerminalNode *Integer();
    antlr4::tree::TerminalNode *RealNumber();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *StringLiteral();
    BuiltInCallContext *builtInCall();
    KernelCallContext *kernelCall();
    SubroutineCallContext *subroutineCall();
    TimingTerminatorContext *timingTerminator();
    antlr4::tree::TerminalNode *MINUS();
    ExpressionTerminatorContext *expressionTerminator();
    antlr4::tree::TerminalNode *LPAREN();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *LBRACKET();
    antlr4::tree::TerminalNode *RBRACKET();
    IncrementorContext *incrementor();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionTerminatorContext* expressionTerminator();
  ExpressionTerminatorContext* expressionTerminator(int precedence);
  class  IncrementorContext : public antlr4::ParserRuleContext {
  public:
    IncrementorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IncrementorContext* incrementor();

  class  BuiltInCallContext : public antlr4::ParserRuleContext {
  public:
    BuiltInCallContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    ExpressionListContext *expressionList();
    antlr4::tree::TerminalNode *RPAREN();
    BuiltInMathContext *builtInMath();
    CastOperatorContext *castOperator();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BuiltInCallContext* builtInCall();

  class  BuiltInMathContext : public antlr4::ParserRuleContext {
  public:
    BuiltInMathContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BuiltInMathContext* builtInMath();

  class  CastOperatorContext : public antlr4::ParserRuleContext {
  public:
    CastOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ClassicalTypeContext *classicalType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CastOperatorContext* castOperator();

  class  ExpressionListContext : public antlr4::ParserRuleContext {
  public:
    ExpressionListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionListContext* expressionList();

  class  BooleanExpressionContext : public antlr4::ParserRuleContext {
  public:
    BooleanExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MembershipTestContext *membershipTest();
    ComparsionExpressionContext *comparsionExpression();
    BooleanExpressionContext *booleanExpression();
    LogicalOperatorContext *logicalOperator();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BooleanExpressionContext* booleanExpression();
  BooleanExpressionContext* booleanExpression(int precedence);
  class  ComparsionExpressionContext : public antlr4::ParserRuleContext {
  public:
    ComparsionExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    RelationalOperatorContext *relationalOperator();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ComparsionExpressionContext* comparsionExpression();

  class  EqualsExpressionContext : public antlr4::ParserRuleContext {
  public:
    EqualsExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EQUALS();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  EqualsExpressionContext* equalsExpression();

  class  AssignmentOperatorContext : public antlr4::ParserRuleContext {
  public:
    AssignmentOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EQUALS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AssignmentOperatorContext* assignmentOperator();

  class  EqualsAssignmentListContext : public antlr4::ParserRuleContext {
  public:
    EqualsAssignmentListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> Identifier();
    antlr4::tree::TerminalNode* Identifier(size_t i);
    std::vector<EqualsExpressionContext *> equalsExpression();
    EqualsExpressionContext* equalsExpression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  EqualsAssignmentListContext* equalsAssignmentList();

  class  MembershipTestContext : public antlr4::ParserRuleContext {
  public:
    MembershipTestContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    SetDeclarationContext *setDeclaration();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  MembershipTestContext* membershipTest();

  class  SetDeclarationContext : public antlr4::ParserRuleContext {
  public:
    SetDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACE();
    ExpressionListContext *expressionList();
    antlr4::tree::TerminalNode *RBRACE();
    RangeDefinitionContext *rangeDefinition();
    antlr4::tree::TerminalNode *Identifier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SetDeclarationContext* setDeclaration();

  class  ProgramBlockContext : public antlr4::ParserRuleContext {
  public:
    ProgramBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);
    antlr4::tree::TerminalNode *LBRACE();
    antlr4::tree::TerminalNode *RBRACE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgramBlockContext* programBlock();

  class  BranchingStatementContext : public antlr4::ParserRuleContext {
  public:
    BranchingStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    BooleanExpressionContext *booleanExpression();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ProgramBlockContext *> programBlock();
    ProgramBlockContext* programBlock(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BranchingStatementContext* branchingStatement();

  class  LoopSignatureContext : public antlr4::ParserRuleContext {
  public:
    LoopSignatureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MembershipTestContext *membershipTest();
    antlr4::tree::TerminalNode *LPAREN();
    BooleanExpressionContext *booleanExpression();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LoopSignatureContext* loopSignature();

  class  LoopStatementContext : public antlr4::ParserRuleContext {
  public:
    LoopStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LoopSignatureContext *loopSignature();
    ProgramBlockContext *programBlock();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LoopStatementContext* loopStatement();

  class  ControlDirectiveStatementContext : public antlr4::ParserRuleContext {
  public:
    ControlDirectiveStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ControlDirectiveContext *controlDirective();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ControlDirectiveStatementContext* controlDirectiveStatement();

  class  ControlDirectiveContext : public antlr4::ParserRuleContext {
  public:
    ControlDirectiveContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ControlDirectiveContext* controlDirective();

  class  KernelDeclarationContext : public antlr4::ParserRuleContext {
  public:
    KernelDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *SEMICOLON();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ReturnSignatureContext *returnSignature();
    ClassicalTypeContext *classicalType();
    ClassicalTypeListContext *classicalTypeList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  KernelDeclarationContext* kernelDeclaration();

  class  KernelCallContext : public antlr4::ParserRuleContext {
  public:
    KernelCallContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ExpressionListContext *expressionList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  KernelCallContext* kernelCall();

  class  SubroutineDefinitionContext : public antlr4::ParserRuleContext {
  public:
    SubroutineDefinitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    SubroutineBlockContext *subroutineBlock();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    QuantumArgumentListContext *quantumArgumentList();
    ReturnSignatureContext *returnSignature();
    ClassicalArgumentListContext *classicalArgumentList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SubroutineDefinitionContext* subroutineDefinition();

  class  ReturnStatementContext : public antlr4::ParserRuleContext {
  public:
    ReturnStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StatementContext *statement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ReturnStatementContext* returnStatement();

  class  SubroutineBlockContext : public antlr4::ParserRuleContext {
  public:
    SubroutineBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACE();
    antlr4::tree::TerminalNode *RBRACE();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);
    ReturnStatementContext *returnStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SubroutineBlockContext* subroutineBlock();

  class  SubroutineCallContext : public antlr4::ParserRuleContext {
  public:
    SubroutineCallContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    std::vector<ExpressionListContext *> expressionList();
    ExpressionListContext* expressionList(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SubroutineCallContext* subroutineCall();

  class  PragmaContext : public antlr4::ParserRuleContext {
  public:
    PragmaContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACE();
    antlr4::tree::TerminalNode *RBRACE();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PragmaContext* pragma();

  class  TimingTypeContext : public antlr4::ParserRuleContext {
  public:
    TimingTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Integer();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TimingTypeContext* timingType();

  class  TimingBoxContext : public antlr4::ParserRuleContext {
  public:
    TimingBoxContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    QuantumBlockContext *quantumBlock();
    antlr4::tree::TerminalNode *TimingLiteral();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TimingBoxContext* timingBox();

  class  TimingTerminatorContext : public antlr4::ParserRuleContext {
  public:
    TimingTerminatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TimingIdentifierContext *timingIdentifier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TimingTerminatorContext* timingTerminator();

  class  TimingIdentifierContext : public antlr4::ParserRuleContext {
  public:
    TimingIdentifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *TimingLiteral();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Identifier();
    QuantumBlockContext *quantumBlock();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TimingIdentifierContext* timingIdentifier();

  class  TimingInstructionNameContext : public antlr4::ParserRuleContext {
  public:
    TimingInstructionNameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TimingInstructionNameContext* timingInstructionName();

  class  TimingInstructionContext : public antlr4::ParserRuleContext {
  public:
    TimingInstructionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TimingInstructionNameContext *timingInstructionName();
    DesignatorContext *designator();
    IndexIdentifierListContext *indexIdentifierList();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ExpressionListContext *expressionList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TimingInstructionContext* timingInstruction();

  class  TimingStatementContext : public antlr4::ParserRuleContext {
  public:
    TimingStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TimingInstructionContext *timingInstruction();
    antlr4::tree::TerminalNode *SEMICOLON();
    TimingBoxContext *timingBox();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TimingStatementContext* timingStatement();

  class  CalibrationContext : public antlr4::ParserRuleContext {
  public:
    CalibrationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    CalibrationGrammarDeclarationContext *calibrationGrammarDeclaration();
    CalibrationDefinitionContext *calibrationDefinition();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CalibrationContext* calibration();

  class  CalibrationGrammarDeclarationContext : public antlr4::ParserRuleContext {
  public:
    CalibrationGrammarDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    CalibrationGrammarContext *calibrationGrammar();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CalibrationGrammarDeclarationContext* calibrationGrammarDeclaration();

  class  CalibrationDefinitionContext : public antlr4::ParserRuleContext {
  public:
    CalibrationDefinitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    IdentifierListContext *identifierList();
    antlr4::tree::TerminalNode *LBRACE();
    antlr4::tree::TerminalNode *RBRACE();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ReturnSignatureContext *returnSignature();
    CalibrationArgumentListContext *calibrationArgumentList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CalibrationDefinitionContext* calibrationDefinition();

  class  CalibrationGrammarContext : public antlr4::ParserRuleContext {
  public:
    CalibrationGrammarContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *StringLiteral();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CalibrationGrammarContext* calibrationGrammar();

  class  CalibrationArgumentListContext : public antlr4::ParserRuleContext {
  public:
    CalibrationArgumentListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ClassicalArgumentListContext *classicalArgumentList();
    ExpressionListContext *expressionList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CalibrationArgumentListContext* calibrationArgumentList();


  virtual bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;
  bool indexIdentifierSempred(IndexIdentifierContext *_localctx, size_t predicateIndex);
  bool expressionSempred(ExpressionContext *_localctx, size_t predicateIndex);
  bool xOrExpressionSempred(XOrExpressionContext *_localctx, size_t predicateIndex);
  bool bitAndExpressionSempred(BitAndExpressionContext *_localctx, size_t predicateIndex);
  bool bitShiftExpressionSempred(BitShiftExpressionContext *_localctx, size_t predicateIndex);
  bool additiveExpressionSempred(AdditiveExpressionContext *_localctx, size_t predicateIndex);
  bool multiplicativeExpressionSempred(MultiplicativeExpressionContext *_localctx, size_t predicateIndex);
  bool expressionTerminatorSempred(ExpressionTerminatorContext *_localctx, size_t predicateIndex);
  bool booleanExpressionSempred(BooleanExpressionContext *_localctx, size_t predicateIndex);

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

}  // namespace qasm3

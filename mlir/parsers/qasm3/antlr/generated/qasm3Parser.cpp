
// Generated from qasm3.g4 by ANTLR 4.9.2


#include "qasm3Listener.h"
#include "qasm3Visitor.h"

#include "qasm3Parser.h"


using namespace antlrcpp;
using namespace qasm3;
using namespace antlr4;

qasm3Parser::qasm3Parser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

qasm3Parser::~qasm3Parser() {
  delete _interpreter;
}

std::string qasm3Parser::getGrammarFileName() const {
  return "qasm3.g4";
}

const std::vector<std::string>& qasm3Parser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& qasm3Parser::getVocabulary() const {
  return _vocabulary;
}


//----------------- ProgramContext ------------------------------------------------------------------

qasm3Parser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::HeaderContext* qasm3Parser::ProgramContext::header() {
  return getRuleContext<qasm3Parser::HeaderContext>(0);
}

std::vector<qasm3Parser::GlobalStatementContext *> qasm3Parser::ProgramContext::globalStatement() {
  return getRuleContexts<qasm3Parser::GlobalStatementContext>();
}

qasm3Parser::GlobalStatementContext* qasm3Parser::ProgramContext::globalStatement(size_t i) {
  return getRuleContext<qasm3Parser::GlobalStatementContext>(i);
}

std::vector<qasm3Parser::StatementContext *> qasm3Parser::ProgramContext::statement() {
  return getRuleContexts<qasm3Parser::StatementContext>();
}

qasm3Parser::StatementContext* qasm3Parser::ProgramContext::statement(size_t i) {
  return getRuleContext<qasm3Parser::StatementContext>(i);
}


size_t qasm3Parser::ProgramContext::getRuleIndex() const {
  return qasm3Parser::RuleProgram;
}

void qasm3Parser::ProgramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgram(this);
}

void qasm3Parser::ProgramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgram(this);
}


antlrcpp::Any qasm3Parser::ProgramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitProgram(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ProgramContext* qasm3Parser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, qasm3Parser::RuleProgram);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(208);
    header();
    setState(213);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__2)
      | (1ULL << qasm3Parser::T__4)
      | (1ULL << qasm3Parser::T__5)
      | (1ULL << qasm3Parser::T__6)
      | (1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__15)
      | (1ULL << qasm3Parser::T__16)
      | (1ULL << qasm3Parser::T__17)
      | (1ULL << qasm3Parser::T__18)
      | (1ULL << qasm3Parser::T__20)
      | (1ULL << qasm3Parser::T__21)
      | (1ULL << qasm3Parser::T__22)
      | (1ULL << qasm3Parser::T__23)
      | (1ULL << qasm3Parser::T__24)
      | (1ULL << qasm3Parser::T__25)
      | (1ULL << qasm3Parser::T__26)
      | (1ULL << qasm3Parser::T__27)
      | (1ULL << qasm3Parser::T__28)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__31)
      | (1ULL << qasm3Parser::T__32)
      | (1ULL << qasm3Parser::T__51)
      | (1ULL << qasm3Parser::T__52)
      | (1ULL << qasm3Parser::T__53)
      | (1ULL << qasm3Parser::T__54)
      | (1ULL << qasm3Parser::T__55)
      | (1ULL << qasm3Parser::T__56)
      | (1ULL << qasm3Parser::T__57)
      | (1ULL << qasm3Parser::T__58)
      | (1ULL << qasm3Parser::T__59)
      | (1ULL << qasm3Parser::T__60)
      | (1ULL << qasm3Parser::T__61)
      | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
      | (1ULL << (qasm3Parser::T__75 - 64))
      | (1ULL << (qasm3Parser::T__77 - 64))
      | (1ULL << (qasm3Parser::T__78 - 64))
      | (1ULL << (qasm3Parser::T__80 - 64))
      | (1ULL << (qasm3Parser::T__81 - 64))
      | (1ULL << (qasm3Parser::T__82 - 64))
      | (1ULL << (qasm3Parser::T__83 - 64))
      | (1ULL << (qasm3Parser::T__84 - 64))
      | (1ULL << (qasm3Parser::T__85 - 64))
      | (1ULL << (qasm3Parser::T__86 - 64))
      | (1ULL << (qasm3Parser::T__87 - 64))
      | (1ULL << (qasm3Parser::T__88 - 64))
      | (1ULL << (qasm3Parser::T__89 - 64))
      | (1ULL << (qasm3Parser::T__90 - 64))
      | (1ULL << (qasm3Parser::T__91 - 64))
      | (1ULL << (qasm3Parser::T__92 - 64))
      | (1ULL << (qasm3Parser::T__93 - 64))
      | (1ULL << (qasm3Parser::T__94 - 64))
      | (1ULL << (qasm3Parser::T__95 - 64))
      | (1ULL << (qasm3Parser::LPAREN - 64))
      | (1ULL << (qasm3Parser::MINUS - 64))
      | (1ULL << (qasm3Parser::Constant - 64))
      | (1ULL << (qasm3Parser::Integer - 64))
      | (1ULL << (qasm3Parser::Identifier - 64))
      | (1ULL << (qasm3Parser::RealNumber - 64))
      | (1ULL << (qasm3Parser::TimingLiteral - 64))
      | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
      setState(211);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case qasm3Parser::T__5:
        case qasm3Parser::T__6:
        case qasm3Parser::T__20:
        case qasm3Parser::T__83:
        case qasm3Parser::T__84:
        case qasm3Parser::T__86:
        case qasm3Parser::T__94:
        case qasm3Parser::T__95: {
          setState(209);
          globalStatement();
          break;
        }

        case qasm3Parser::T__2:
        case qasm3Parser::T__4:
        case qasm3Parser::T__7:
        case qasm3Parser::T__8:
        case qasm3Parser::T__9:
        case qasm3Parser::T__10:
        case qasm3Parser::T__11:
        case qasm3Parser::T__12:
        case qasm3Parser::T__13:
        case qasm3Parser::T__14:
        case qasm3Parser::T__15:
        case qasm3Parser::T__16:
        case qasm3Parser::T__17:
        case qasm3Parser::T__18:
        case qasm3Parser::T__21:
        case qasm3Parser::T__22:
        case qasm3Parser::T__23:
        case qasm3Parser::T__24:
        case qasm3Parser::T__25:
        case qasm3Parser::T__26:
        case qasm3Parser::T__27:
        case qasm3Parser::T__28:
        case qasm3Parser::T__30:
        case qasm3Parser::T__31:
        case qasm3Parser::T__32:
        case qasm3Parser::T__51:
        case qasm3Parser::T__52:
        case qasm3Parser::T__53:
        case qasm3Parser::T__54:
        case qasm3Parser::T__55:
        case qasm3Parser::T__56:
        case qasm3Parser::T__57:
        case qasm3Parser::T__58:
        case qasm3Parser::T__59:
        case qasm3Parser::T__60:
        case qasm3Parser::T__61:
        case qasm3Parser::T__62:
        case qasm3Parser::T__63:
        case qasm3Parser::T__75:
        case qasm3Parser::T__77:
        case qasm3Parser::T__78:
        case qasm3Parser::T__80:
        case qasm3Parser::T__81:
        case qasm3Parser::T__82:
        case qasm3Parser::T__85:
        case qasm3Parser::T__87:
        case qasm3Parser::T__88:
        case qasm3Parser::T__89:
        case qasm3Parser::T__90:
        case qasm3Parser::T__91:
        case qasm3Parser::T__92:
        case qasm3Parser::T__93:
        case qasm3Parser::LPAREN:
        case qasm3Parser::MINUS:
        case qasm3Parser::Constant:
        case qasm3Parser::Integer:
        case qasm3Parser::Identifier:
        case qasm3Parser::RealNumber:
        case qasm3Parser::TimingLiteral:
        case qasm3Parser::StringLiteral: {
          setState(210);
          statement();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(215);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- HeaderContext ------------------------------------------------------------------

qasm3Parser::HeaderContext::HeaderContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::VersionContext* qasm3Parser::HeaderContext::version() {
  return getRuleContext<qasm3Parser::VersionContext>(0);
}

std::vector<qasm3Parser::IncludeContext *> qasm3Parser::HeaderContext::include() {
  return getRuleContexts<qasm3Parser::IncludeContext>();
}

qasm3Parser::IncludeContext* qasm3Parser::HeaderContext::include(size_t i) {
  return getRuleContext<qasm3Parser::IncludeContext>(i);
}


size_t qasm3Parser::HeaderContext::getRuleIndex() const {
  return qasm3Parser::RuleHeader;
}

void qasm3Parser::HeaderContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterHeader(this);
}

void qasm3Parser::HeaderContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitHeader(this);
}


antlrcpp::Any qasm3Parser::HeaderContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitHeader(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::HeaderContext* qasm3Parser::header() {
  HeaderContext *_localctx = _tracker.createInstance<HeaderContext>(_ctx, getState());
  enterRule(_localctx, 2, qasm3Parser::RuleHeader);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(217);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::T__0) {
      setState(216);
      version();
    }
    setState(222);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == qasm3Parser::T__1) {
      setState(219);
      include();
      setState(224);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VersionContext ------------------------------------------------------------------

qasm3Parser::VersionContext::VersionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::VersionContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}

tree::TerminalNode* qasm3Parser::VersionContext::Integer() {
  return getToken(qasm3Parser::Integer, 0);
}

tree::TerminalNode* qasm3Parser::VersionContext::RealNumber() {
  return getToken(qasm3Parser::RealNumber, 0);
}


size_t qasm3Parser::VersionContext::getRuleIndex() const {
  return qasm3Parser::RuleVersion;
}

void qasm3Parser::VersionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVersion(this);
}

void qasm3Parser::VersionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVersion(this);
}


antlrcpp::Any qasm3Parser::VersionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitVersion(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::VersionContext* qasm3Parser::version() {
  VersionContext *_localctx = _tracker.createInstance<VersionContext>(_ctx, getState());
  enterRule(_localctx, 4, qasm3Parser::RuleVersion);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(225);
    match(qasm3Parser::T__0);
    setState(226);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::Integer

    || _la == qasm3Parser::RealNumber)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(227);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IncludeContext ------------------------------------------------------------------

qasm3Parser::IncludeContext::IncludeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::IncludeContext::StringLiteral() {
  return getToken(qasm3Parser::StringLiteral, 0);
}

tree::TerminalNode* qasm3Parser::IncludeContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::IncludeContext::getRuleIndex() const {
  return qasm3Parser::RuleInclude;
}

void qasm3Parser::IncludeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInclude(this);
}

void qasm3Parser::IncludeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInclude(this);
}


antlrcpp::Any qasm3Parser::IncludeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitInclude(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::IncludeContext* qasm3Parser::include() {
  IncludeContext *_localctx = _tracker.createInstance<IncludeContext>(_ctx, getState());
  enterRule(_localctx, 6, qasm3Parser::RuleInclude);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(229);
    match(qasm3Parser::T__1);
    setState(230);
    match(qasm3Parser::StringLiteral);
    setState(231);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GlobalStatementContext ------------------------------------------------------------------

qasm3Parser::GlobalStatementContext::GlobalStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::SubroutineDefinitionContext* qasm3Parser::GlobalStatementContext::subroutineDefinition() {
  return getRuleContext<qasm3Parser::SubroutineDefinitionContext>(0);
}

qasm3Parser::KernelDeclarationContext* qasm3Parser::GlobalStatementContext::kernelDeclaration() {
  return getRuleContext<qasm3Parser::KernelDeclarationContext>(0);
}

qasm3Parser::QuantumGateDefinitionContext* qasm3Parser::GlobalStatementContext::quantumGateDefinition() {
  return getRuleContext<qasm3Parser::QuantumGateDefinitionContext>(0);
}

qasm3Parser::CalibrationContext* qasm3Parser::GlobalStatementContext::calibration() {
  return getRuleContext<qasm3Parser::CalibrationContext>(0);
}

qasm3Parser::QuantumDeclarationStatementContext* qasm3Parser::GlobalStatementContext::quantumDeclarationStatement() {
  return getRuleContext<qasm3Parser::QuantumDeclarationStatementContext>(0);
}

qasm3Parser::PragmaContext* qasm3Parser::GlobalStatementContext::pragma() {
  return getRuleContext<qasm3Parser::PragmaContext>(0);
}


size_t qasm3Parser::GlobalStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleGlobalStatement;
}

void qasm3Parser::GlobalStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGlobalStatement(this);
}

void qasm3Parser::GlobalStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGlobalStatement(this);
}


antlrcpp::Any qasm3Parser::GlobalStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitGlobalStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::GlobalStatementContext* qasm3Parser::globalStatement() {
  GlobalStatementContext *_localctx = _tracker.createInstance<GlobalStatementContext>(_ctx, getState());
  enterRule(_localctx, 8, qasm3Parser::RuleGlobalStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(239);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__84: {
        enterOuterAlt(_localctx, 1);
        setState(233);
        subroutineDefinition();
        break;
      }

      case qasm3Parser::T__83: {
        enterOuterAlt(_localctx, 2);
        setState(234);
        kernelDeclaration();
        break;
      }

      case qasm3Parser::T__20: {
        enterOuterAlt(_localctx, 3);
        setState(235);
        quantumGateDefinition();
        break;
      }

      case qasm3Parser::T__94:
      case qasm3Parser::T__95: {
        enterOuterAlt(_localctx, 4);
        setState(236);
        calibration();
        break;
      }

      case qasm3Parser::T__5:
      case qasm3Parser::T__6: {
        enterOuterAlt(_localctx, 5);
        setState(237);
        quantumDeclarationStatement();
        break;
      }

      case qasm3Parser::T__86: {
        enterOuterAlt(_localctx, 6);
        setState(238);
        pragma();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

qasm3Parser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ExpressionStatementContext* qasm3Parser::StatementContext::expressionStatement() {
  return getRuleContext<qasm3Parser::ExpressionStatementContext>(0);
}

qasm3Parser::AssignmentStatementContext* qasm3Parser::StatementContext::assignmentStatement() {
  return getRuleContext<qasm3Parser::AssignmentStatementContext>(0);
}

qasm3Parser::ClassicalDeclarationStatementContext* qasm3Parser::StatementContext::classicalDeclarationStatement() {
  return getRuleContext<qasm3Parser::ClassicalDeclarationStatementContext>(0);
}

qasm3Parser::BranchingStatementContext* qasm3Parser::StatementContext::branchingStatement() {
  return getRuleContext<qasm3Parser::BranchingStatementContext>(0);
}

qasm3Parser::LoopStatementContext* qasm3Parser::StatementContext::loopStatement() {
  return getRuleContext<qasm3Parser::LoopStatementContext>(0);
}

qasm3Parser::ControlDirectiveStatementContext* qasm3Parser::StatementContext::controlDirectiveStatement() {
  return getRuleContext<qasm3Parser::ControlDirectiveStatementContext>(0);
}

qasm3Parser::AliasStatementContext* qasm3Parser::StatementContext::aliasStatement() {
  return getRuleContext<qasm3Parser::AliasStatementContext>(0);
}

qasm3Parser::QuantumStatementContext* qasm3Parser::StatementContext::quantumStatement() {
  return getRuleContext<qasm3Parser::QuantumStatementContext>(0);
}

qasm3Parser::ReturnStatementContext* qasm3Parser::StatementContext::returnStatement() {
  return getRuleContext<qasm3Parser::ReturnStatementContext>(0);
}

qasm3Parser::Qcor_test_statementContext* qasm3Parser::StatementContext::qcor_test_statement() {
  return getRuleContext<qasm3Parser::Qcor_test_statementContext>(0);
}

qasm3Parser::Compute_action_stmtContext* qasm3Parser::StatementContext::compute_action_stmt() {
  return getRuleContext<qasm3Parser::Compute_action_stmtContext>(0);
}


size_t qasm3Parser::StatementContext::getRuleIndex() const {
  return qasm3Parser::RuleStatement;
}

void qasm3Parser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void qasm3Parser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}


antlrcpp::Any qasm3Parser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::StatementContext* qasm3Parser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 10, qasm3Parser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(252);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(241);
      expressionStatement();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(242);
      assignmentStatement();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(243);
      classicalDeclarationStatement();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(244);
      branchingStatement();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(245);
      loopStatement();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(246);
      controlDirectiveStatement();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(247);
      aliasStatement();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(248);
      quantumStatement();
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(249);
      returnStatement();
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(250);
      qcor_test_statement();
      break;
    }

    case 11: {
      enterOuterAlt(_localctx, 11);
      setState(251);
      compute_action_stmt();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Compute_action_stmtContext ------------------------------------------------------------------

qasm3Parser::Compute_action_stmtContext::Compute_action_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::ProgramBlockContext *> qasm3Parser::Compute_action_stmtContext::programBlock() {
  return getRuleContexts<qasm3Parser::ProgramBlockContext>();
}

qasm3Parser::ProgramBlockContext* qasm3Parser::Compute_action_stmtContext::programBlock(size_t i) {
  return getRuleContext<qasm3Parser::ProgramBlockContext>(i);
}


size_t qasm3Parser::Compute_action_stmtContext::getRuleIndex() const {
  return qasm3Parser::RuleCompute_action_stmt;
}

void qasm3Parser::Compute_action_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompute_action_stmt(this);
}

void qasm3Parser::Compute_action_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompute_action_stmt(this);
}


antlrcpp::Any qasm3Parser::Compute_action_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCompute_action_stmt(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::Compute_action_stmtContext* qasm3Parser::compute_action_stmt() {
  Compute_action_stmtContext *_localctx = _tracker.createInstance<Compute_action_stmtContext>(_ctx, getState());
  enterRule(_localctx, 12, qasm3Parser::RuleCompute_action_stmt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(254);
    match(qasm3Parser::T__2);
    setState(255);
    dynamic_cast<Compute_action_stmtContext *>(_localctx)->compute_block = programBlock();
    setState(256);
    match(qasm3Parser::T__3);
    setState(257);
    dynamic_cast<Compute_action_stmtContext *>(_localctx)->action_block = programBlock();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Qcor_test_statementContext ------------------------------------------------------------------

qasm3Parser::Qcor_test_statementContext::Qcor_test_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::Qcor_test_statementContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

qasm3Parser::BooleanExpressionContext* qasm3Parser::Qcor_test_statementContext::booleanExpression() {
  return getRuleContext<qasm3Parser::BooleanExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::Qcor_test_statementContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

tree::TerminalNode* qasm3Parser::Qcor_test_statementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::Qcor_test_statementContext::getRuleIndex() const {
  return qasm3Parser::RuleQcor_test_statement;
}

void qasm3Parser::Qcor_test_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQcor_test_statement(this);
}

void qasm3Parser::Qcor_test_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQcor_test_statement(this);
}


antlrcpp::Any qasm3Parser::Qcor_test_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQcor_test_statement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::Qcor_test_statementContext* qasm3Parser::qcor_test_statement() {
  Qcor_test_statementContext *_localctx = _tracker.createInstance<Qcor_test_statementContext>(_ctx, getState());
  enterRule(_localctx, 14, qasm3Parser::RuleQcor_test_statement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(259);
    match(qasm3Parser::T__4);
    setState(260);
    match(qasm3Parser::LPAREN);
    setState(261);
    booleanExpression(0);
    setState(262);
    match(qasm3Parser::RPAREN);
    setState(263);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumDeclarationStatementContext ------------------------------------------------------------------

qasm3Parser::QuantumDeclarationStatementContext::QuantumDeclarationStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::QuantumDeclarationContext* qasm3Parser::QuantumDeclarationStatementContext::quantumDeclaration() {
  return getRuleContext<qasm3Parser::QuantumDeclarationContext>(0);
}

tree::TerminalNode* qasm3Parser::QuantumDeclarationStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::QuantumDeclarationStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumDeclarationStatement;
}

void qasm3Parser::QuantumDeclarationStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumDeclarationStatement(this);
}

void qasm3Parser::QuantumDeclarationStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumDeclarationStatement(this);
}


antlrcpp::Any qasm3Parser::QuantumDeclarationStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumDeclarationStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumDeclarationStatementContext* qasm3Parser::quantumDeclarationStatement() {
  QuantumDeclarationStatementContext *_localctx = _tracker.createInstance<QuantumDeclarationStatementContext>(_ctx, getState());
  enterRule(_localctx, 16, qasm3Parser::RuleQuantumDeclarationStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(265);
    quantumDeclaration();
    setState(266);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalDeclarationStatementContext ------------------------------------------------------------------

qasm3Parser::ClassicalDeclarationStatementContext::ClassicalDeclarationStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::ClassicalDeclarationStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}

qasm3Parser::ClassicalDeclarationContext* qasm3Parser::ClassicalDeclarationStatementContext::classicalDeclaration() {
  return getRuleContext<qasm3Parser::ClassicalDeclarationContext>(0);
}

qasm3Parser::ConstantDeclarationContext* qasm3Parser::ClassicalDeclarationStatementContext::constantDeclaration() {
  return getRuleContext<qasm3Parser::ConstantDeclarationContext>(0);
}


size_t qasm3Parser::ClassicalDeclarationStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleClassicalDeclarationStatement;
}

void qasm3Parser::ClassicalDeclarationStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalDeclarationStatement(this);
}

void qasm3Parser::ClassicalDeclarationStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalDeclarationStatement(this);
}


antlrcpp::Any qasm3Parser::ClassicalDeclarationStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitClassicalDeclarationStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ClassicalDeclarationStatementContext* qasm3Parser::classicalDeclarationStatement() {
  ClassicalDeclarationStatementContext *_localctx = _tracker.createInstance<ClassicalDeclarationStatementContext>(_ctx, getState());
  enterRule(_localctx, 18, qasm3Parser::RuleClassicalDeclarationStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(270);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__7:
      case qasm3Parser::T__8:
      case qasm3Parser::T__9:
      case qasm3Parser::T__10:
      case qasm3Parser::T__11:
      case qasm3Parser::T__12:
      case qasm3Parser::T__13:
      case qasm3Parser::T__14:
      case qasm3Parser::T__15:
      case qasm3Parser::T__16:
      case qasm3Parser::T__87:
      case qasm3Parser::T__88: {
        setState(268);
        classicalDeclaration();
        break;
      }

      case qasm3Parser::T__17: {
        setState(269);
        constantDeclaration();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(272);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalAssignmentContext ------------------------------------------------------------------

qasm3Parser::ClassicalAssignmentContext::ClassicalAssignmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::IndexIdentifierContext *> qasm3Parser::ClassicalAssignmentContext::indexIdentifier() {
  return getRuleContexts<qasm3Parser::IndexIdentifierContext>();
}

qasm3Parser::IndexIdentifierContext* qasm3Parser::ClassicalAssignmentContext::indexIdentifier(size_t i) {
  return getRuleContext<qasm3Parser::IndexIdentifierContext>(i);
}

qasm3Parser::AssignmentOperatorContext* qasm3Parser::ClassicalAssignmentContext::assignmentOperator() {
  return getRuleContext<qasm3Parser::AssignmentOperatorContext>(0);
}

qasm3Parser::ExpressionContext* qasm3Parser::ClassicalAssignmentContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}


size_t qasm3Parser::ClassicalAssignmentContext::getRuleIndex() const {
  return qasm3Parser::RuleClassicalAssignment;
}

void qasm3Parser::ClassicalAssignmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalAssignment(this);
}

void qasm3Parser::ClassicalAssignmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalAssignment(this);
}


antlrcpp::Any qasm3Parser::ClassicalAssignmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitClassicalAssignment(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ClassicalAssignmentContext* qasm3Parser::classicalAssignment() {
  ClassicalAssignmentContext *_localctx = _tracker.createInstance<ClassicalAssignmentContext>(_ctx, getState());
  enterRule(_localctx, 20, qasm3Parser::RuleClassicalAssignment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(274);
    indexIdentifier(0);
    setState(275);
    assignmentOperator();
    setState(278);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      setState(276);
      expression(0);
      break;
    }

    case 2: {
      setState(277);
      indexIdentifier(0);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentStatementContext ------------------------------------------------------------------

qasm3Parser::AssignmentStatementContext::AssignmentStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::AssignmentStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}

qasm3Parser::ClassicalAssignmentContext* qasm3Parser::AssignmentStatementContext::classicalAssignment() {
  return getRuleContext<qasm3Parser::ClassicalAssignmentContext>(0);
}

qasm3Parser::QuantumMeasurementAssignmentContext* qasm3Parser::AssignmentStatementContext::quantumMeasurementAssignment() {
  return getRuleContext<qasm3Parser::QuantumMeasurementAssignmentContext>(0);
}


size_t qasm3Parser::AssignmentStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleAssignmentStatement;
}

void qasm3Parser::AssignmentStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignmentStatement(this);
}

void qasm3Parser::AssignmentStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignmentStatement(this);
}


antlrcpp::Any qasm3Parser::AssignmentStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitAssignmentStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::AssignmentStatementContext* qasm3Parser::assignmentStatement() {
  AssignmentStatementContext *_localctx = _tracker.createInstance<AssignmentStatementContext>(_ctx, getState());
  enterRule(_localctx, 22, qasm3Parser::RuleAssignmentStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(282);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      setState(280);
      classicalAssignment();
      break;
    }

    case 2: {
      setState(281);
      quantumMeasurementAssignment();
      break;
    }

    default:
      break;
    }
    setState(284);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReturnSignatureContext ------------------------------------------------------------------

qasm3Parser::ReturnSignatureContext::ReturnSignatureContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::ReturnSignatureContext::ARROW() {
  return getToken(qasm3Parser::ARROW, 0);
}

qasm3Parser::ClassicalTypeContext* qasm3Parser::ReturnSignatureContext::classicalType() {
  return getRuleContext<qasm3Parser::ClassicalTypeContext>(0);
}


size_t qasm3Parser::ReturnSignatureContext::getRuleIndex() const {
  return qasm3Parser::RuleReturnSignature;
}

void qasm3Parser::ReturnSignatureContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReturnSignature(this);
}

void qasm3Parser::ReturnSignatureContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReturnSignature(this);
}


antlrcpp::Any qasm3Parser::ReturnSignatureContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitReturnSignature(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ReturnSignatureContext* qasm3Parser::returnSignature() {
  ReturnSignatureContext *_localctx = _tracker.createInstance<ReturnSignatureContext>(_ctx, getState());
  enterRule(_localctx, 24, qasm3Parser::RuleReturnSignature);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(286);
    match(qasm3Parser::ARROW);
    setState(287);
    classicalType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DesignatorContext ------------------------------------------------------------------

qasm3Parser::DesignatorContext::DesignatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::DesignatorContext::LBRACKET() {
  return getToken(qasm3Parser::LBRACKET, 0);
}

qasm3Parser::ExpressionContext* qasm3Parser::DesignatorContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::DesignatorContext::RBRACKET() {
  return getToken(qasm3Parser::RBRACKET, 0);
}


size_t qasm3Parser::DesignatorContext::getRuleIndex() const {
  return qasm3Parser::RuleDesignator;
}

void qasm3Parser::DesignatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDesignator(this);
}

void qasm3Parser::DesignatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDesignator(this);
}


antlrcpp::Any qasm3Parser::DesignatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitDesignator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::DesignatorContext* qasm3Parser::designator() {
  DesignatorContext *_localctx = _tracker.createInstance<DesignatorContext>(_ctx, getState());
  enterRule(_localctx, 26, qasm3Parser::RuleDesignator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(289);
    match(qasm3Parser::LBRACKET);
    setState(290);
    expression(0);
    setState(291);
    match(qasm3Parser::RBRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DoubleDesignatorContext ------------------------------------------------------------------

qasm3Parser::DoubleDesignatorContext::DoubleDesignatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::DoubleDesignatorContext::LBRACKET() {
  return getToken(qasm3Parser::LBRACKET, 0);
}

std::vector<qasm3Parser::ExpressionContext *> qasm3Parser::DoubleDesignatorContext::expression() {
  return getRuleContexts<qasm3Parser::ExpressionContext>();
}

qasm3Parser::ExpressionContext* qasm3Parser::DoubleDesignatorContext::expression(size_t i) {
  return getRuleContext<qasm3Parser::ExpressionContext>(i);
}

tree::TerminalNode* qasm3Parser::DoubleDesignatorContext::COMMA() {
  return getToken(qasm3Parser::COMMA, 0);
}

tree::TerminalNode* qasm3Parser::DoubleDesignatorContext::RBRACKET() {
  return getToken(qasm3Parser::RBRACKET, 0);
}


size_t qasm3Parser::DoubleDesignatorContext::getRuleIndex() const {
  return qasm3Parser::RuleDoubleDesignator;
}

void qasm3Parser::DoubleDesignatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDoubleDesignator(this);
}

void qasm3Parser::DoubleDesignatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDoubleDesignator(this);
}


antlrcpp::Any qasm3Parser::DoubleDesignatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitDoubleDesignator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::DoubleDesignatorContext* qasm3Parser::doubleDesignator() {
  DoubleDesignatorContext *_localctx = _tracker.createInstance<DoubleDesignatorContext>(_ctx, getState());
  enterRule(_localctx, 28, qasm3Parser::RuleDoubleDesignator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(293);
    match(qasm3Parser::LBRACKET);
    setState(294);
    expression(0);
    setState(295);
    match(qasm3Parser::COMMA);
    setState(296);
    expression(0);
    setState(297);
    match(qasm3Parser::RBRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifierListContext ------------------------------------------------------------------

qasm3Parser::IdentifierListContext::IdentifierListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> qasm3Parser::IdentifierListContext::Identifier() {
  return getTokens(qasm3Parser::Identifier);
}

tree::TerminalNode* qasm3Parser::IdentifierListContext::Identifier(size_t i) {
  return getToken(qasm3Parser::Identifier, i);
}

std::vector<tree::TerminalNode *> qasm3Parser::IdentifierListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::IdentifierListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::IdentifierListContext::getRuleIndex() const {
  return qasm3Parser::RuleIdentifierList;
}

void qasm3Parser::IdentifierListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdentifierList(this);
}

void qasm3Parser::IdentifierListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdentifierList(this);
}


antlrcpp::Any qasm3Parser::IdentifierListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitIdentifierList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::IdentifierListContext* qasm3Parser::identifierList() {
  IdentifierListContext *_localctx = _tracker.createInstance<IdentifierListContext>(_ctx, getState());
  enterRule(_localctx, 30, qasm3Parser::RuleIdentifierList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(303);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(299);
        match(qasm3Parser::Identifier);
        setState(300);
        match(qasm3Parser::COMMA); 
      }
      setState(305);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    }
    setState(306);
    match(qasm3Parser::Identifier);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssociationContext ------------------------------------------------------------------

qasm3Parser::AssociationContext::AssociationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::AssociationContext::COLON() {
  return getToken(qasm3Parser::COLON, 0);
}

tree::TerminalNode* qasm3Parser::AssociationContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}


size_t qasm3Parser::AssociationContext::getRuleIndex() const {
  return qasm3Parser::RuleAssociation;
}

void qasm3Parser::AssociationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssociation(this);
}

void qasm3Parser::AssociationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssociation(this);
}


antlrcpp::Any qasm3Parser::AssociationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitAssociation(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::AssociationContext* qasm3Parser::association() {
  AssociationContext *_localctx = _tracker.createInstance<AssociationContext>(_ctx, getState());
  enterRule(_localctx, 32, qasm3Parser::RuleAssociation);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(308);
    match(qasm3Parser::COLON);
    setState(309);
    match(qasm3Parser::Identifier);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumTypeContext ------------------------------------------------------------------

qasm3Parser::QuantumTypeContext::QuantumTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::QuantumTypeContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumType;
}

void qasm3Parser::QuantumTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumType(this);
}

void qasm3Parser::QuantumTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumType(this);
}


antlrcpp::Any qasm3Parser::QuantumTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumType(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumTypeContext* qasm3Parser::quantumType() {
  QuantumTypeContext *_localctx = _tracker.createInstance<QuantumTypeContext>(_ctx, getState());
  enterRule(_localctx, 34, qasm3Parser::RuleQuantumType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(311);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__5

    || _la == qasm3Parser::T__6)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumDeclarationContext ------------------------------------------------------------------

qasm3Parser::QuantumDeclarationContext::QuantumDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::QuantumDeclarationContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::DesignatorContext* qasm3Parser::QuantumDeclarationContext::designator() {
  return getRuleContext<qasm3Parser::DesignatorContext>(0);
}


size_t qasm3Parser::QuantumDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumDeclaration;
}

void qasm3Parser::QuantumDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumDeclaration(this);
}

void qasm3Parser::QuantumDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumDeclaration(this);
}


antlrcpp::Any qasm3Parser::QuantumDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumDeclarationContext* qasm3Parser::quantumDeclaration() {
  QuantumDeclarationContext *_localctx = _tracker.createInstance<QuantumDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 36, qasm3Parser::RuleQuantumDeclaration);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(323);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__6: {
        enterOuterAlt(_localctx, 1);
        setState(313);
        match(qasm3Parser::T__6);
        setState(314);
        match(qasm3Parser::Identifier);
        setState(316);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == qasm3Parser::LBRACKET) {
          setState(315);
          designator();
        }
        break;
      }

      case qasm3Parser::T__5: {
        enterOuterAlt(_localctx, 2);
        setState(318);
        match(qasm3Parser::T__5);
        setState(320);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == qasm3Parser::LBRACKET) {
          setState(319);
          designator();
        }
        setState(322);
        match(qasm3Parser::Identifier);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumArgumentContext ------------------------------------------------------------------

qasm3Parser::QuantumArgumentContext::QuantumArgumentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::QuantumTypeContext* qasm3Parser::QuantumArgumentContext::quantumType() {
  return getRuleContext<qasm3Parser::QuantumTypeContext>(0);
}

qasm3Parser::AssociationContext* qasm3Parser::QuantumArgumentContext::association() {
  return getRuleContext<qasm3Parser::AssociationContext>(0);
}

qasm3Parser::DesignatorContext* qasm3Parser::QuantumArgumentContext::designator() {
  return getRuleContext<qasm3Parser::DesignatorContext>(0);
}


size_t qasm3Parser::QuantumArgumentContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumArgument;
}

void qasm3Parser::QuantumArgumentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumArgument(this);
}

void qasm3Parser::QuantumArgumentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumArgument(this);
}


antlrcpp::Any qasm3Parser::QuantumArgumentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumArgument(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumArgumentContext* qasm3Parser::quantumArgument() {
  QuantumArgumentContext *_localctx = _tracker.createInstance<QuantumArgumentContext>(_ctx, getState());
  enterRule(_localctx, 38, qasm3Parser::RuleQuantumArgument);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(325);
    quantumType();
    setState(327);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LBRACKET) {
      setState(326);
      designator();
    }
    setState(329);
    association();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumArgumentListContext ------------------------------------------------------------------

qasm3Parser::QuantumArgumentListContext::QuantumArgumentListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::QuantumArgumentContext *> qasm3Parser::QuantumArgumentListContext::quantumArgument() {
  return getRuleContexts<qasm3Parser::QuantumArgumentContext>();
}

qasm3Parser::QuantumArgumentContext* qasm3Parser::QuantumArgumentListContext::quantumArgument(size_t i) {
  return getRuleContext<qasm3Parser::QuantumArgumentContext>(i);
}

std::vector<tree::TerminalNode *> qasm3Parser::QuantumArgumentListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::QuantumArgumentListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::QuantumArgumentListContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumArgumentList;
}

void qasm3Parser::QuantumArgumentListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumArgumentList(this);
}

void qasm3Parser::QuantumArgumentListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumArgumentList(this);
}


antlrcpp::Any qasm3Parser::QuantumArgumentListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumArgumentList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumArgumentListContext* qasm3Parser::quantumArgumentList() {
  QuantumArgumentListContext *_localctx = _tracker.createInstance<QuantumArgumentListContext>(_ctx, getState());
  enterRule(_localctx, 40, qasm3Parser::RuleQuantumArgumentList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(336);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(331);
        quantumArgument();
        setState(332);
        match(qasm3Parser::COMMA); 
      }
      setState(338);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx);
    }
    setState(339);
    quantumArgument();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BitTypeContext ------------------------------------------------------------------

qasm3Parser::BitTypeContext::BitTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::BitTypeContext::getRuleIndex() const {
  return qasm3Parser::RuleBitType;
}

void qasm3Parser::BitTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBitType(this);
}

void qasm3Parser::BitTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBitType(this);
}


antlrcpp::Any qasm3Parser::BitTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBitType(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::BitTypeContext* qasm3Parser::bitType() {
  BitTypeContext *_localctx = _tracker.createInstance<BitTypeContext>(_ctx, getState());
  enterRule(_localctx, 42, qasm3Parser::RuleBitType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(341);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__7

    || _la == qasm3Parser::T__8)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SingleDesignatorTypeContext ------------------------------------------------------------------

qasm3Parser::SingleDesignatorTypeContext::SingleDesignatorTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::SingleDesignatorTypeContext::getRuleIndex() const {
  return qasm3Parser::RuleSingleDesignatorType;
}

void qasm3Parser::SingleDesignatorTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingleDesignatorType(this);
}

void qasm3Parser::SingleDesignatorTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingleDesignatorType(this);
}


antlrcpp::Any qasm3Parser::SingleDesignatorTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitSingleDesignatorType(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::SingleDesignatorTypeContext* qasm3Parser::singleDesignatorType() {
  SingleDesignatorTypeContext *_localctx = _tracker.createInstance<SingleDesignatorTypeContext>(_ctx, getState());
  enterRule(_localctx, 44, qasm3Parser::RuleSingleDesignatorType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(343);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DoubleDesignatorTypeContext ------------------------------------------------------------------

qasm3Parser::DoubleDesignatorTypeContext::DoubleDesignatorTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::DoubleDesignatorTypeContext::getRuleIndex() const {
  return qasm3Parser::RuleDoubleDesignatorType;
}

void qasm3Parser::DoubleDesignatorTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDoubleDesignatorType(this);
}

void qasm3Parser::DoubleDesignatorTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDoubleDesignatorType(this);
}


antlrcpp::Any qasm3Parser::DoubleDesignatorTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitDoubleDesignatorType(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::DoubleDesignatorTypeContext* qasm3Parser::doubleDesignatorType() {
  DoubleDesignatorTypeContext *_localctx = _tracker.createInstance<DoubleDesignatorTypeContext>(_ctx, getState());
  enterRule(_localctx, 46, qasm3Parser::RuleDoubleDesignatorType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(345);
    match(qasm3Parser::T__13);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NoDesignatorTypeContext ------------------------------------------------------------------

qasm3Parser::NoDesignatorTypeContext::NoDesignatorTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::TimingTypeContext* qasm3Parser::NoDesignatorTypeContext::timingType() {
  return getRuleContext<qasm3Parser::TimingTypeContext>(0);
}


size_t qasm3Parser::NoDesignatorTypeContext::getRuleIndex() const {
  return qasm3Parser::RuleNoDesignatorType;
}

void qasm3Parser::NoDesignatorTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNoDesignatorType(this);
}

void qasm3Parser::NoDesignatorTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNoDesignatorType(this);
}


antlrcpp::Any qasm3Parser::NoDesignatorTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitNoDesignatorType(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::NoDesignatorTypeContext* qasm3Parser::noDesignatorType() {
  NoDesignatorTypeContext *_localctx = _tracker.createInstance<NoDesignatorTypeContext>(_ctx, getState());
  enterRule(_localctx, 48, qasm3Parser::RuleNoDesignatorType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(353);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__14: {
        enterOuterAlt(_localctx, 1);
        setState(347);
        match(qasm3Parser::T__14);
        break;
      }

      case qasm3Parser::T__87:
      case qasm3Parser::T__88: {
        enterOuterAlt(_localctx, 2);
        setState(348);
        timingType();
        break;
      }

      case qasm3Parser::T__9: {
        enterOuterAlt(_localctx, 3);
        setState(349);
        match(qasm3Parser::T__9);
        break;
      }

      case qasm3Parser::T__15: {
        enterOuterAlt(_localctx, 4);
        setState(350);
        match(qasm3Parser::T__15);
        break;
      }

      case qasm3Parser::T__11: {
        enterOuterAlt(_localctx, 5);
        setState(351);
        match(qasm3Parser::T__11);
        break;
      }

      case qasm3Parser::T__16: {
        enterOuterAlt(_localctx, 6);
        setState(352);
        match(qasm3Parser::T__16);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalTypeContext ------------------------------------------------------------------

qasm3Parser::ClassicalTypeContext::ClassicalTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::SingleDesignatorTypeContext* qasm3Parser::ClassicalTypeContext::singleDesignatorType() {
  return getRuleContext<qasm3Parser::SingleDesignatorTypeContext>(0);
}

qasm3Parser::DesignatorContext* qasm3Parser::ClassicalTypeContext::designator() {
  return getRuleContext<qasm3Parser::DesignatorContext>(0);
}

qasm3Parser::DoubleDesignatorTypeContext* qasm3Parser::ClassicalTypeContext::doubleDesignatorType() {
  return getRuleContext<qasm3Parser::DoubleDesignatorTypeContext>(0);
}

qasm3Parser::DoubleDesignatorContext* qasm3Parser::ClassicalTypeContext::doubleDesignator() {
  return getRuleContext<qasm3Parser::DoubleDesignatorContext>(0);
}

qasm3Parser::NoDesignatorTypeContext* qasm3Parser::ClassicalTypeContext::noDesignatorType() {
  return getRuleContext<qasm3Parser::NoDesignatorTypeContext>(0);
}

qasm3Parser::BitTypeContext* qasm3Parser::ClassicalTypeContext::bitType() {
  return getRuleContext<qasm3Parser::BitTypeContext>(0);
}


size_t qasm3Parser::ClassicalTypeContext::getRuleIndex() const {
  return qasm3Parser::RuleClassicalType;
}

void qasm3Parser::ClassicalTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalType(this);
}

void qasm3Parser::ClassicalTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalType(this);
}


antlrcpp::Any qasm3Parser::ClassicalTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitClassicalType(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ClassicalTypeContext* qasm3Parser::classicalType() {
  ClassicalTypeContext *_localctx = _tracker.createInstance<ClassicalTypeContext>(_ctx, getState());
  enterRule(_localctx, 50, qasm3Parser::RuleClassicalType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(366);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(355);
      singleDesignatorType();
      setState(356);
      designator();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(358);
      doubleDesignatorType();
      setState(359);
      doubleDesignator();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(361);
      noDesignatorType();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(362);
      bitType();
      setState(364);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == qasm3Parser::LBRACKET) {
        setState(363);
        designator();
      }
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstantDeclarationContext ------------------------------------------------------------------

qasm3Parser::ConstantDeclarationContext::ConstantDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::EqualsAssignmentListContext* qasm3Parser::ConstantDeclarationContext::equalsAssignmentList() {
  return getRuleContext<qasm3Parser::EqualsAssignmentListContext>(0);
}


size_t qasm3Parser::ConstantDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleConstantDeclaration;
}

void qasm3Parser::ConstantDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstantDeclaration(this);
}

void qasm3Parser::ConstantDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstantDeclaration(this);
}


antlrcpp::Any qasm3Parser::ConstantDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitConstantDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ConstantDeclarationContext* qasm3Parser::constantDeclaration() {
  ConstantDeclarationContext *_localctx = _tracker.createInstance<ConstantDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 52, qasm3Parser::RuleConstantDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(368);
    match(qasm3Parser::T__17);
    setState(369);
    equalsAssignmentList();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SingleDesignatorDeclarationContext ------------------------------------------------------------------

qasm3Parser::SingleDesignatorDeclarationContext::SingleDesignatorDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::SingleDesignatorTypeContext* qasm3Parser::SingleDesignatorDeclarationContext::singleDesignatorType() {
  return getRuleContext<qasm3Parser::SingleDesignatorTypeContext>(0);
}

qasm3Parser::DesignatorContext* qasm3Parser::SingleDesignatorDeclarationContext::designator() {
  return getRuleContext<qasm3Parser::DesignatorContext>(0);
}

qasm3Parser::IdentifierListContext* qasm3Parser::SingleDesignatorDeclarationContext::identifierList() {
  return getRuleContext<qasm3Parser::IdentifierListContext>(0);
}

qasm3Parser::EqualsAssignmentListContext* qasm3Parser::SingleDesignatorDeclarationContext::equalsAssignmentList() {
  return getRuleContext<qasm3Parser::EqualsAssignmentListContext>(0);
}


size_t qasm3Parser::SingleDesignatorDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleSingleDesignatorDeclaration;
}

void qasm3Parser::SingleDesignatorDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingleDesignatorDeclaration(this);
}

void qasm3Parser::SingleDesignatorDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingleDesignatorDeclaration(this);
}


antlrcpp::Any qasm3Parser::SingleDesignatorDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitSingleDesignatorDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::SingleDesignatorDeclarationContext* qasm3Parser::singleDesignatorDeclaration() {
  SingleDesignatorDeclarationContext *_localctx = _tracker.createInstance<SingleDesignatorDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 54, qasm3Parser::RuleSingleDesignatorDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(371);
    singleDesignatorType();
    setState(372);
    designator();
    setState(375);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
    case 1: {
      setState(373);
      identifierList();
      break;
    }

    case 2: {
      setState(374);
      equalsAssignmentList();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DoubleDesignatorDeclarationContext ------------------------------------------------------------------

qasm3Parser::DoubleDesignatorDeclarationContext::DoubleDesignatorDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::DoubleDesignatorTypeContext* qasm3Parser::DoubleDesignatorDeclarationContext::doubleDesignatorType() {
  return getRuleContext<qasm3Parser::DoubleDesignatorTypeContext>(0);
}

qasm3Parser::DoubleDesignatorContext* qasm3Parser::DoubleDesignatorDeclarationContext::doubleDesignator() {
  return getRuleContext<qasm3Parser::DoubleDesignatorContext>(0);
}

qasm3Parser::IdentifierListContext* qasm3Parser::DoubleDesignatorDeclarationContext::identifierList() {
  return getRuleContext<qasm3Parser::IdentifierListContext>(0);
}

qasm3Parser::EqualsAssignmentListContext* qasm3Parser::DoubleDesignatorDeclarationContext::equalsAssignmentList() {
  return getRuleContext<qasm3Parser::EqualsAssignmentListContext>(0);
}


size_t qasm3Parser::DoubleDesignatorDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleDoubleDesignatorDeclaration;
}

void qasm3Parser::DoubleDesignatorDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDoubleDesignatorDeclaration(this);
}

void qasm3Parser::DoubleDesignatorDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDoubleDesignatorDeclaration(this);
}


antlrcpp::Any qasm3Parser::DoubleDesignatorDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitDoubleDesignatorDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::DoubleDesignatorDeclarationContext* qasm3Parser::doubleDesignatorDeclaration() {
  DoubleDesignatorDeclarationContext *_localctx = _tracker.createInstance<DoubleDesignatorDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 56, qasm3Parser::RuleDoubleDesignatorDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(377);
    doubleDesignatorType();
    setState(378);
    doubleDesignator();
    setState(381);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
    case 1: {
      setState(379);
      identifierList();
      break;
    }

    case 2: {
      setState(380);
      equalsAssignmentList();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NoDesignatorDeclarationContext ------------------------------------------------------------------

qasm3Parser::NoDesignatorDeclarationContext::NoDesignatorDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::NoDesignatorTypeContext* qasm3Parser::NoDesignatorDeclarationContext::noDesignatorType() {
  return getRuleContext<qasm3Parser::NoDesignatorTypeContext>(0);
}

qasm3Parser::IdentifierListContext* qasm3Parser::NoDesignatorDeclarationContext::identifierList() {
  return getRuleContext<qasm3Parser::IdentifierListContext>(0);
}

qasm3Parser::EqualsAssignmentListContext* qasm3Parser::NoDesignatorDeclarationContext::equalsAssignmentList() {
  return getRuleContext<qasm3Parser::EqualsAssignmentListContext>(0);
}


size_t qasm3Parser::NoDesignatorDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleNoDesignatorDeclaration;
}

void qasm3Parser::NoDesignatorDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNoDesignatorDeclaration(this);
}

void qasm3Parser::NoDesignatorDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNoDesignatorDeclaration(this);
}


antlrcpp::Any qasm3Parser::NoDesignatorDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitNoDesignatorDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::NoDesignatorDeclarationContext* qasm3Parser::noDesignatorDeclaration() {
  NoDesignatorDeclarationContext *_localctx = _tracker.createInstance<NoDesignatorDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 58, qasm3Parser::RuleNoDesignatorDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(383);
    noDesignatorType();
    setState(386);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx)) {
    case 1: {
      setState(384);
      identifierList();
      break;
    }

    case 2: {
      setState(385);
      equalsAssignmentList();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BitDeclarationContext ------------------------------------------------------------------

qasm3Parser::BitDeclarationContext::BitDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::BitTypeContext* qasm3Parser::BitDeclarationContext::bitType() {
  return getRuleContext<qasm3Parser::BitTypeContext>(0);
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::BitDeclarationContext::indexIdentifierList() {
  return getRuleContext<qasm3Parser::IndexIdentifierListContext>(0);
}

qasm3Parser::IndexEqualsAssignmentListContext* qasm3Parser::BitDeclarationContext::indexEqualsAssignmentList() {
  return getRuleContext<qasm3Parser::IndexEqualsAssignmentListContext>(0);
}


size_t qasm3Parser::BitDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleBitDeclaration;
}

void qasm3Parser::BitDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBitDeclaration(this);
}

void qasm3Parser::BitDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBitDeclaration(this);
}


antlrcpp::Any qasm3Parser::BitDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBitDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::BitDeclarationContext* qasm3Parser::bitDeclaration() {
  BitDeclarationContext *_localctx = _tracker.createInstance<BitDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 60, qasm3Parser::RuleBitDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(388);
    bitType();
    setState(391);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx)) {
    case 1: {
      setState(389);
      indexIdentifierList();
      break;
    }

    case 2: {
      setState(390);
      indexEqualsAssignmentList();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalDeclarationContext ------------------------------------------------------------------

qasm3Parser::ClassicalDeclarationContext::ClassicalDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::SingleDesignatorDeclarationContext* qasm3Parser::ClassicalDeclarationContext::singleDesignatorDeclaration() {
  return getRuleContext<qasm3Parser::SingleDesignatorDeclarationContext>(0);
}

qasm3Parser::DoubleDesignatorDeclarationContext* qasm3Parser::ClassicalDeclarationContext::doubleDesignatorDeclaration() {
  return getRuleContext<qasm3Parser::DoubleDesignatorDeclarationContext>(0);
}

qasm3Parser::NoDesignatorDeclarationContext* qasm3Parser::ClassicalDeclarationContext::noDesignatorDeclaration() {
  return getRuleContext<qasm3Parser::NoDesignatorDeclarationContext>(0);
}

qasm3Parser::BitDeclarationContext* qasm3Parser::ClassicalDeclarationContext::bitDeclaration() {
  return getRuleContext<qasm3Parser::BitDeclarationContext>(0);
}


size_t qasm3Parser::ClassicalDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleClassicalDeclaration;
}

void qasm3Parser::ClassicalDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalDeclaration(this);
}

void qasm3Parser::ClassicalDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalDeclaration(this);
}


antlrcpp::Any qasm3Parser::ClassicalDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitClassicalDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ClassicalDeclarationContext* qasm3Parser::classicalDeclaration() {
  ClassicalDeclarationContext *_localctx = _tracker.createInstance<ClassicalDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 62, qasm3Parser::RuleClassicalDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(397);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 22, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(393);
      singleDesignatorDeclaration();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(394);
      doubleDesignatorDeclaration();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(395);
      noDesignatorDeclaration();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(396);
      bitDeclaration();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalTypeListContext ------------------------------------------------------------------

qasm3Parser::ClassicalTypeListContext::ClassicalTypeListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::ClassicalTypeContext *> qasm3Parser::ClassicalTypeListContext::classicalType() {
  return getRuleContexts<qasm3Parser::ClassicalTypeContext>();
}

qasm3Parser::ClassicalTypeContext* qasm3Parser::ClassicalTypeListContext::classicalType(size_t i) {
  return getRuleContext<qasm3Parser::ClassicalTypeContext>(i);
}

std::vector<tree::TerminalNode *> qasm3Parser::ClassicalTypeListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::ClassicalTypeListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::ClassicalTypeListContext::getRuleIndex() const {
  return qasm3Parser::RuleClassicalTypeList;
}

void qasm3Parser::ClassicalTypeListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalTypeList(this);
}

void qasm3Parser::ClassicalTypeListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalTypeList(this);
}


antlrcpp::Any qasm3Parser::ClassicalTypeListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitClassicalTypeList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ClassicalTypeListContext* qasm3Parser::classicalTypeList() {
  ClassicalTypeListContext *_localctx = _tracker.createInstance<ClassicalTypeListContext>(_ctx, getState());
  enterRule(_localctx, 64, qasm3Parser::RuleClassicalTypeList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(404);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(399);
        classicalType();
        setState(400);
        match(qasm3Parser::COMMA); 
      }
      setState(406);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx);
    }
    setState(407);
    classicalType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalArgumentContext ------------------------------------------------------------------

qasm3Parser::ClassicalArgumentContext::ClassicalArgumentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ClassicalTypeContext* qasm3Parser::ClassicalArgumentContext::classicalType() {
  return getRuleContext<qasm3Parser::ClassicalTypeContext>(0);
}

qasm3Parser::AssociationContext* qasm3Parser::ClassicalArgumentContext::association() {
  return getRuleContext<qasm3Parser::AssociationContext>(0);
}


size_t qasm3Parser::ClassicalArgumentContext::getRuleIndex() const {
  return qasm3Parser::RuleClassicalArgument;
}

void qasm3Parser::ClassicalArgumentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalArgument(this);
}

void qasm3Parser::ClassicalArgumentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalArgument(this);
}


antlrcpp::Any qasm3Parser::ClassicalArgumentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitClassicalArgument(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ClassicalArgumentContext* qasm3Parser::classicalArgument() {
  ClassicalArgumentContext *_localctx = _tracker.createInstance<ClassicalArgumentContext>(_ctx, getState());
  enterRule(_localctx, 66, qasm3Parser::RuleClassicalArgument);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(409);
    classicalType();
    setState(410);
    association();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalArgumentListContext ------------------------------------------------------------------

qasm3Parser::ClassicalArgumentListContext::ClassicalArgumentListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::ClassicalArgumentContext *> qasm3Parser::ClassicalArgumentListContext::classicalArgument() {
  return getRuleContexts<qasm3Parser::ClassicalArgumentContext>();
}

qasm3Parser::ClassicalArgumentContext* qasm3Parser::ClassicalArgumentListContext::classicalArgument(size_t i) {
  return getRuleContext<qasm3Parser::ClassicalArgumentContext>(i);
}

std::vector<tree::TerminalNode *> qasm3Parser::ClassicalArgumentListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::ClassicalArgumentListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::ClassicalArgumentListContext::getRuleIndex() const {
  return qasm3Parser::RuleClassicalArgumentList;
}

void qasm3Parser::ClassicalArgumentListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalArgumentList(this);
}

void qasm3Parser::ClassicalArgumentListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalArgumentList(this);
}


antlrcpp::Any qasm3Parser::ClassicalArgumentListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitClassicalArgumentList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ClassicalArgumentListContext* qasm3Parser::classicalArgumentList() {
  ClassicalArgumentListContext *_localctx = _tracker.createInstance<ClassicalArgumentListContext>(_ctx, getState());
  enterRule(_localctx, 68, qasm3Parser::RuleClassicalArgumentList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(417);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(412);
        classicalArgument();
        setState(413);
        match(qasm3Parser::COMMA); 
      }
      setState(419);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    }
    setState(420);
    classicalArgument();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AliasStatementContext ------------------------------------------------------------------

qasm3Parser::AliasStatementContext::AliasStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::AliasStatementContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

tree::TerminalNode* qasm3Parser::AliasStatementContext::EQUALS() {
  return getToken(qasm3Parser::EQUALS, 0);
}

qasm3Parser::IndexIdentifierContext* qasm3Parser::AliasStatementContext::indexIdentifier() {
  return getRuleContext<qasm3Parser::IndexIdentifierContext>(0);
}

tree::TerminalNode* qasm3Parser::AliasStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::AliasStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleAliasStatement;
}

void qasm3Parser::AliasStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAliasStatement(this);
}

void qasm3Parser::AliasStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAliasStatement(this);
}


antlrcpp::Any qasm3Parser::AliasStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitAliasStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::AliasStatementContext* qasm3Parser::aliasStatement() {
  AliasStatementContext *_localctx = _tracker.createInstance<AliasStatementContext>(_ctx, getState());
  enterRule(_localctx, 70, qasm3Parser::RuleAliasStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(422);
    match(qasm3Parser::T__18);
    setState(423);
    match(qasm3Parser::Identifier);
    setState(424);
    match(qasm3Parser::EQUALS);
    setState(425);
    indexIdentifier(0);
    setState(426);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IndexIdentifierContext ------------------------------------------------------------------

qasm3Parser::IndexIdentifierContext::IndexIdentifierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::IndexIdentifierContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::RangeDefinitionContext* qasm3Parser::IndexIdentifierContext::rangeDefinition() {
  return getRuleContext<qasm3Parser::RangeDefinitionContext>(0);
}

tree::TerminalNode* qasm3Parser::IndexIdentifierContext::LBRACKET() {
  return getToken(qasm3Parser::LBRACKET, 0);
}

qasm3Parser::ExpressionListContext* qasm3Parser::IndexIdentifierContext::expressionList() {
  return getRuleContext<qasm3Parser::ExpressionListContext>(0);
}

tree::TerminalNode* qasm3Parser::IndexIdentifierContext::RBRACKET() {
  return getToken(qasm3Parser::RBRACKET, 0);
}

std::vector<qasm3Parser::IndexIdentifierContext *> qasm3Parser::IndexIdentifierContext::indexIdentifier() {
  return getRuleContexts<qasm3Parser::IndexIdentifierContext>();
}

qasm3Parser::IndexIdentifierContext* qasm3Parser::IndexIdentifierContext::indexIdentifier(size_t i) {
  return getRuleContext<qasm3Parser::IndexIdentifierContext>(i);
}


size_t qasm3Parser::IndexIdentifierContext::getRuleIndex() const {
  return qasm3Parser::RuleIndexIdentifier;
}

void qasm3Parser::IndexIdentifierContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexIdentifier(this);
}

void qasm3Parser::IndexIdentifierContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexIdentifier(this);
}


antlrcpp::Any qasm3Parser::IndexIdentifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitIndexIdentifier(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::IndexIdentifierContext* qasm3Parser::indexIdentifier() {
   return indexIdentifier(0);
}

qasm3Parser::IndexIdentifierContext* qasm3Parser::indexIdentifier(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::IndexIdentifierContext *_localctx = _tracker.createInstance<IndexIdentifierContext>(_ctx, parentState);
  qasm3Parser::IndexIdentifierContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 72;
  enterRecursionRule(_localctx, 72, qasm3Parser::RuleIndexIdentifier, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(438);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 26, _ctx)) {
    case 1: {
      setState(429);
      match(qasm3Parser::Identifier);
      setState(430);
      rangeDefinition();
      break;
    }

    case 2: {
      setState(431);
      match(qasm3Parser::Identifier);
      setState(436);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx)) {
      case 1: {
        setState(432);
        match(qasm3Parser::LBRACKET);
        setState(433);
        expressionList();
        setState(434);
        match(qasm3Parser::RBRACKET);
        break;
      }

      default:
        break;
      }
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(445);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 27, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<IndexIdentifierContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleIndexIdentifier);
        setState(440);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(441);
        match(qasm3Parser::T__19);
        setState(442);
        indexIdentifier(2); 
      }
      setState(447);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 27, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- IndexIdentifierListContext ------------------------------------------------------------------

qasm3Parser::IndexIdentifierListContext::IndexIdentifierListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::IndexIdentifierContext *> qasm3Parser::IndexIdentifierListContext::indexIdentifier() {
  return getRuleContexts<qasm3Parser::IndexIdentifierContext>();
}

qasm3Parser::IndexIdentifierContext* qasm3Parser::IndexIdentifierListContext::indexIdentifier(size_t i) {
  return getRuleContext<qasm3Parser::IndexIdentifierContext>(i);
}

std::vector<tree::TerminalNode *> qasm3Parser::IndexIdentifierListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::IndexIdentifierListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::IndexIdentifierListContext::getRuleIndex() const {
  return qasm3Parser::RuleIndexIdentifierList;
}

void qasm3Parser::IndexIdentifierListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexIdentifierList(this);
}

void qasm3Parser::IndexIdentifierListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexIdentifierList(this);
}


antlrcpp::Any qasm3Parser::IndexIdentifierListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitIndexIdentifierList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::indexIdentifierList() {
  IndexIdentifierListContext *_localctx = _tracker.createInstance<IndexIdentifierListContext>(_ctx, getState());
  enterRule(_localctx, 74, qasm3Parser::RuleIndexIdentifierList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(453);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 28, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(448);
        indexIdentifier(0);
        setState(449);
        match(qasm3Parser::COMMA); 
      }
      setState(455);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 28, _ctx);
    }
    setState(456);
    indexIdentifier(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IndexEqualsAssignmentListContext ------------------------------------------------------------------

qasm3Parser::IndexEqualsAssignmentListContext::IndexEqualsAssignmentListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::IndexIdentifierContext *> qasm3Parser::IndexEqualsAssignmentListContext::indexIdentifier() {
  return getRuleContexts<qasm3Parser::IndexIdentifierContext>();
}

qasm3Parser::IndexIdentifierContext* qasm3Parser::IndexEqualsAssignmentListContext::indexIdentifier(size_t i) {
  return getRuleContext<qasm3Parser::IndexIdentifierContext>(i);
}

std::vector<qasm3Parser::EqualsExpressionContext *> qasm3Parser::IndexEqualsAssignmentListContext::equalsExpression() {
  return getRuleContexts<qasm3Parser::EqualsExpressionContext>();
}

qasm3Parser::EqualsExpressionContext* qasm3Parser::IndexEqualsAssignmentListContext::equalsExpression(size_t i) {
  return getRuleContext<qasm3Parser::EqualsExpressionContext>(i);
}

std::vector<tree::TerminalNode *> qasm3Parser::IndexEqualsAssignmentListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::IndexEqualsAssignmentListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::IndexEqualsAssignmentListContext::getRuleIndex() const {
  return qasm3Parser::RuleIndexEqualsAssignmentList;
}

void qasm3Parser::IndexEqualsAssignmentListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexEqualsAssignmentList(this);
}

void qasm3Parser::IndexEqualsAssignmentListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexEqualsAssignmentList(this);
}


antlrcpp::Any qasm3Parser::IndexEqualsAssignmentListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitIndexEqualsAssignmentList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::IndexEqualsAssignmentListContext* qasm3Parser::indexEqualsAssignmentList() {
  IndexEqualsAssignmentListContext *_localctx = _tracker.createInstance<IndexEqualsAssignmentListContext>(_ctx, getState());
  enterRule(_localctx, 76, qasm3Parser::RuleIndexEqualsAssignmentList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(464);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 29, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(458);
        indexIdentifier(0);
        setState(459);
        equalsExpression();
        setState(460);
        match(qasm3Parser::COMMA); 
      }
      setState(466);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 29, _ctx);
    }
    setState(467);
    indexIdentifier(0);
    setState(468);
    equalsExpression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RangeDefinitionContext ------------------------------------------------------------------

qasm3Parser::RangeDefinitionContext::RangeDefinitionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::RangeDefinitionContext::LBRACKET() {
  return getToken(qasm3Parser::LBRACKET, 0);
}

std::vector<tree::TerminalNode *> qasm3Parser::RangeDefinitionContext::COLON() {
  return getTokens(qasm3Parser::COLON);
}

tree::TerminalNode* qasm3Parser::RangeDefinitionContext::COLON(size_t i) {
  return getToken(qasm3Parser::COLON, i);
}

tree::TerminalNode* qasm3Parser::RangeDefinitionContext::RBRACKET() {
  return getToken(qasm3Parser::RBRACKET, 0);
}

std::vector<qasm3Parser::ExpressionContext *> qasm3Parser::RangeDefinitionContext::expression() {
  return getRuleContexts<qasm3Parser::ExpressionContext>();
}

qasm3Parser::ExpressionContext* qasm3Parser::RangeDefinitionContext::expression(size_t i) {
  return getRuleContext<qasm3Parser::ExpressionContext>(i);
}


size_t qasm3Parser::RangeDefinitionContext::getRuleIndex() const {
  return qasm3Parser::RuleRangeDefinition;
}

void qasm3Parser::RangeDefinitionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRangeDefinition(this);
}

void qasm3Parser::RangeDefinitionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRangeDefinition(this);
}


antlrcpp::Any qasm3Parser::RangeDefinitionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitRangeDefinition(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::RangeDefinitionContext* qasm3Parser::rangeDefinition() {
  RangeDefinitionContext *_localctx = _tracker.createInstance<RangeDefinitionContext>(_ctx, getState());
  enterRule(_localctx, 78, qasm3Parser::RuleRangeDefinition);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(470);
    match(qasm3Parser::LBRACKET);
    setState(472);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__15)
      | (1ULL << qasm3Parser::T__16)
      | (1ULL << qasm3Parser::T__31)
      | (1ULL << qasm3Parser::T__32)
      | (1ULL << qasm3Parser::T__51)
      | (1ULL << qasm3Parser::T__52)
      | (1ULL << qasm3Parser::T__53)
      | (1ULL << qasm3Parser::T__54)
      | (1ULL << qasm3Parser::T__55)
      | (1ULL << qasm3Parser::T__56)
      | (1ULL << qasm3Parser::T__57)
      | (1ULL << qasm3Parser::T__58)
      | (1ULL << qasm3Parser::T__59)
      | (1ULL << qasm3Parser::T__60)
      | (1ULL << qasm3Parser::T__61)
      | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
      | (1ULL << (qasm3Parser::T__87 - 64))
      | (1ULL << (qasm3Parser::T__88 - 64))
      | (1ULL << (qasm3Parser::T__91 - 64))
      | (1ULL << (qasm3Parser::LPAREN - 64))
      | (1ULL << (qasm3Parser::MINUS - 64))
      | (1ULL << (qasm3Parser::Constant - 64))
      | (1ULL << (qasm3Parser::Integer - 64))
      | (1ULL << (qasm3Parser::Identifier - 64))
      | (1ULL << (qasm3Parser::RealNumber - 64))
      | (1ULL << (qasm3Parser::TimingLiteral - 64))
      | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
      setState(471);
      expression(0);
    }
    setState(474);
    match(qasm3Parser::COLON);
    setState(476);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__15)
      | (1ULL << qasm3Parser::T__16)
      | (1ULL << qasm3Parser::T__31)
      | (1ULL << qasm3Parser::T__32)
      | (1ULL << qasm3Parser::T__51)
      | (1ULL << qasm3Parser::T__52)
      | (1ULL << qasm3Parser::T__53)
      | (1ULL << qasm3Parser::T__54)
      | (1ULL << qasm3Parser::T__55)
      | (1ULL << qasm3Parser::T__56)
      | (1ULL << qasm3Parser::T__57)
      | (1ULL << qasm3Parser::T__58)
      | (1ULL << qasm3Parser::T__59)
      | (1ULL << qasm3Parser::T__60)
      | (1ULL << qasm3Parser::T__61)
      | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
      | (1ULL << (qasm3Parser::T__87 - 64))
      | (1ULL << (qasm3Parser::T__88 - 64))
      | (1ULL << (qasm3Parser::T__91 - 64))
      | (1ULL << (qasm3Parser::LPAREN - 64))
      | (1ULL << (qasm3Parser::MINUS - 64))
      | (1ULL << (qasm3Parser::Constant - 64))
      | (1ULL << (qasm3Parser::Integer - 64))
      | (1ULL << (qasm3Parser::Identifier - 64))
      | (1ULL << (qasm3Parser::RealNumber - 64))
      | (1ULL << (qasm3Parser::TimingLiteral - 64))
      | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
      setState(475);
      expression(0);
    }
    setState(480);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::COLON) {
      setState(478);
      match(qasm3Parser::COLON);
      setState(479);
      expression(0);
    }
    setState(482);
    match(qasm3Parser::RBRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumGateDefinitionContext ------------------------------------------------------------------

qasm3Parser::QuantumGateDefinitionContext::QuantumGateDefinitionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::QuantumGateSignatureContext* qasm3Parser::QuantumGateDefinitionContext::quantumGateSignature() {
  return getRuleContext<qasm3Parser::QuantumGateSignatureContext>(0);
}

qasm3Parser::QuantumBlockContext* qasm3Parser::QuantumGateDefinitionContext::quantumBlock() {
  return getRuleContext<qasm3Parser::QuantumBlockContext>(0);
}


size_t qasm3Parser::QuantumGateDefinitionContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumGateDefinition;
}

void qasm3Parser::QuantumGateDefinitionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumGateDefinition(this);
}

void qasm3Parser::QuantumGateDefinitionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumGateDefinition(this);
}


antlrcpp::Any qasm3Parser::QuantumGateDefinitionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumGateDefinition(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumGateDefinitionContext* qasm3Parser::quantumGateDefinition() {
  QuantumGateDefinitionContext *_localctx = _tracker.createInstance<QuantumGateDefinitionContext>(_ctx, getState());
  enterRule(_localctx, 80, qasm3Parser::RuleQuantumGateDefinition);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(484);
    match(qasm3Parser::T__20);
    setState(485);
    quantumGateSignature();
    setState(486);
    quantumBlock();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumGateSignatureContext ------------------------------------------------------------------

qasm3Parser::QuantumGateSignatureContext::QuantumGateSignatureContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::IdentifierListContext *> qasm3Parser::QuantumGateSignatureContext::identifierList() {
  return getRuleContexts<qasm3Parser::IdentifierListContext>();
}

qasm3Parser::IdentifierListContext* qasm3Parser::QuantumGateSignatureContext::identifierList(size_t i) {
  return getRuleContext<qasm3Parser::IdentifierListContext>(i);
}

tree::TerminalNode* qasm3Parser::QuantumGateSignatureContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

tree::TerminalNode* qasm3Parser::QuantumGateSignatureContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::QuantumGateSignatureContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}


size_t qasm3Parser::QuantumGateSignatureContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumGateSignature;
}

void qasm3Parser::QuantumGateSignatureContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumGateSignature(this);
}

void qasm3Parser::QuantumGateSignatureContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumGateSignature(this);
}


antlrcpp::Any qasm3Parser::QuantumGateSignatureContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumGateSignature(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumGateSignatureContext* qasm3Parser::quantumGateSignature() {
  QuantumGateSignatureContext *_localctx = _tracker.createInstance<QuantumGateSignatureContext>(_ctx, getState());
  enterRule(_localctx, 82, qasm3Parser::RuleQuantumGateSignature);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(488);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__21

    || _la == qasm3Parser::T__22 || _la == qasm3Parser::Identifier)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(494);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(489);
      match(qasm3Parser::LPAREN);
      setState(491);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == qasm3Parser::Identifier) {
        setState(490);
        identifierList();
      }
      setState(493);
      match(qasm3Parser::RPAREN);
    }
    setState(496);
    identifierList();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumBlockContext ------------------------------------------------------------------

qasm3Parser::QuantumBlockContext::QuantumBlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::QuantumBlockContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

tree::TerminalNode* qasm3Parser::QuantumBlockContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}

std::vector<qasm3Parser::Compute_action_stmtContext *> qasm3Parser::QuantumBlockContext::compute_action_stmt() {
  return getRuleContexts<qasm3Parser::Compute_action_stmtContext>();
}

qasm3Parser::Compute_action_stmtContext* qasm3Parser::QuantumBlockContext::compute_action_stmt(size_t i) {
  return getRuleContext<qasm3Parser::Compute_action_stmtContext>(i);
}

std::vector<qasm3Parser::QuantumStatementContext *> qasm3Parser::QuantumBlockContext::quantumStatement() {
  return getRuleContexts<qasm3Parser::QuantumStatementContext>();
}

qasm3Parser::QuantumStatementContext* qasm3Parser::QuantumBlockContext::quantumStatement(size_t i) {
  return getRuleContext<qasm3Parser::QuantumStatementContext>(i);
}

std::vector<qasm3Parser::QuantumLoopContext *> qasm3Parser::QuantumBlockContext::quantumLoop() {
  return getRuleContexts<qasm3Parser::QuantumLoopContext>();
}

qasm3Parser::QuantumLoopContext* qasm3Parser::QuantumBlockContext::quantumLoop(size_t i) {
  return getRuleContext<qasm3Parser::QuantumLoopContext>(i);
}


size_t qasm3Parser::QuantumBlockContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumBlock;
}

void qasm3Parser::QuantumBlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumBlock(this);
}

void qasm3Parser::QuantumBlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumBlock(this);
}


antlrcpp::Any qasm3Parser::QuantumBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumBlock(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumBlockContext* qasm3Parser::quantumBlock() {
  QuantumBlockContext *_localctx = _tracker.createInstance<QuantumBlockContext>(_ctx, getState());
  enterRule(_localctx, 84, qasm3Parser::RuleQuantumBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(498);
    match(qasm3Parser::LBRACE);
    setState(504);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__2)
      | (1ULL << qasm3Parser::T__21)
      | (1ULL << qasm3Parser::T__22)
      | (1ULL << qasm3Parser::T__23)
      | (1ULL << qasm3Parser::T__24)
      | (1ULL << qasm3Parser::T__25)
      | (1ULL << qasm3Parser::T__26)
      | (1ULL << qasm3Parser::T__27)
      | (1ULL << qasm3Parser::T__28)
      | (1ULL << qasm3Parser::T__30))) != 0) || ((((_la - 78) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 78)) & ((1ULL << (qasm3Parser::T__77 - 78))
      | (1ULL << (qasm3Parser::T__78 - 78))
      | (1ULL << (qasm3Parser::T__89 - 78))
      | (1ULL << (qasm3Parser::T__90 - 78))
      | (1ULL << (qasm3Parser::T__92 - 78))
      | (1ULL << (qasm3Parser::T__93 - 78))
      | (1ULL << (qasm3Parser::Identifier - 78)))) != 0)) {
      setState(502);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case qasm3Parser::T__2: {
          setState(499);
          compute_action_stmt();
          break;
        }

        case qasm3Parser::T__21:
        case qasm3Parser::T__22:
        case qasm3Parser::T__23:
        case qasm3Parser::T__24:
        case qasm3Parser::T__25:
        case qasm3Parser::T__26:
        case qasm3Parser::T__27:
        case qasm3Parser::T__28:
        case qasm3Parser::T__30:
        case qasm3Parser::T__89:
        case qasm3Parser::T__90:
        case qasm3Parser::T__92:
        case qasm3Parser::T__93:
        case qasm3Parser::Identifier: {
          setState(500);
          quantumStatement();
          break;
        }

        case qasm3Parser::T__77:
        case qasm3Parser::T__78: {
          setState(501);
          quantumLoop();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(506);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(507);
    match(qasm3Parser::RBRACE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumLoopContext ------------------------------------------------------------------

qasm3Parser::QuantumLoopContext::QuantumLoopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::LoopSignatureContext* qasm3Parser::QuantumLoopContext::loopSignature() {
  return getRuleContext<qasm3Parser::LoopSignatureContext>(0);
}

qasm3Parser::QuantumLoopBlockContext* qasm3Parser::QuantumLoopContext::quantumLoopBlock() {
  return getRuleContext<qasm3Parser::QuantumLoopBlockContext>(0);
}


size_t qasm3Parser::QuantumLoopContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumLoop;
}

void qasm3Parser::QuantumLoopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumLoop(this);
}

void qasm3Parser::QuantumLoopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumLoop(this);
}


antlrcpp::Any qasm3Parser::QuantumLoopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumLoop(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumLoopContext* qasm3Parser::quantumLoop() {
  QuantumLoopContext *_localctx = _tracker.createInstance<QuantumLoopContext>(_ctx, getState());
  enterRule(_localctx, 86, qasm3Parser::RuleQuantumLoop);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(509);
    loopSignature();
    setState(510);
    quantumLoopBlock();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumLoopBlockContext ------------------------------------------------------------------

qasm3Parser::QuantumLoopBlockContext::QuantumLoopBlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::QuantumStatementContext *> qasm3Parser::QuantumLoopBlockContext::quantumStatement() {
  return getRuleContexts<qasm3Parser::QuantumStatementContext>();
}

qasm3Parser::QuantumStatementContext* qasm3Parser::QuantumLoopBlockContext::quantumStatement(size_t i) {
  return getRuleContext<qasm3Parser::QuantumStatementContext>(i);
}

tree::TerminalNode* qasm3Parser::QuantumLoopBlockContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

tree::TerminalNode* qasm3Parser::QuantumLoopBlockContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}


size_t qasm3Parser::QuantumLoopBlockContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumLoopBlock;
}

void qasm3Parser::QuantumLoopBlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumLoopBlock(this);
}

void qasm3Parser::QuantumLoopBlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumLoopBlock(this);
}


antlrcpp::Any qasm3Parser::QuantumLoopBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumLoopBlock(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumLoopBlockContext* qasm3Parser::quantumLoopBlock() {
  QuantumLoopBlockContext *_localctx = _tracker.createInstance<QuantumLoopBlockContext>(_ctx, getState());
  enterRule(_localctx, 88, qasm3Parser::RuleQuantumLoopBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(521);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__21:
      case qasm3Parser::T__22:
      case qasm3Parser::T__23:
      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26:
      case qasm3Parser::T__27:
      case qasm3Parser::T__28:
      case qasm3Parser::T__30:
      case qasm3Parser::T__89:
      case qasm3Parser::T__90:
      case qasm3Parser::T__92:
      case qasm3Parser::T__93:
      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(512);
        quantumStatement();
        break;
      }

      case qasm3Parser::LBRACE: {
        enterOuterAlt(_localctx, 2);
        setState(513);
        match(qasm3Parser::LBRACE);
        setState(517);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << qasm3Parser::T__21)
          | (1ULL << qasm3Parser::T__22)
          | (1ULL << qasm3Parser::T__23)
          | (1ULL << qasm3Parser::T__24)
          | (1ULL << qasm3Parser::T__25)
          | (1ULL << qasm3Parser::T__26)
          | (1ULL << qasm3Parser::T__27)
          | (1ULL << qasm3Parser::T__28)
          | (1ULL << qasm3Parser::T__30))) != 0) || ((((_la - 90) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 90)) & ((1ULL << (qasm3Parser::T__89 - 90))
          | (1ULL << (qasm3Parser::T__90 - 90))
          | (1ULL << (qasm3Parser::T__92 - 90))
          | (1ULL << (qasm3Parser::T__93 - 90))
          | (1ULL << (qasm3Parser::Identifier - 90)))) != 0)) {
          setState(514);
          quantumStatement();
          setState(519);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(520);
        match(qasm3Parser::RBRACE);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumStatementContext ------------------------------------------------------------------

qasm3Parser::QuantumStatementContext::QuantumStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::QuantumInstructionContext* qasm3Parser::QuantumStatementContext::quantumInstruction() {
  return getRuleContext<qasm3Parser::QuantumInstructionContext>(0);
}

tree::TerminalNode* qasm3Parser::QuantumStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}

qasm3Parser::TimingStatementContext* qasm3Parser::QuantumStatementContext::timingStatement() {
  return getRuleContext<qasm3Parser::TimingStatementContext>(0);
}


size_t qasm3Parser::QuantumStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumStatement;
}

void qasm3Parser::QuantumStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumStatement(this);
}

void qasm3Parser::QuantumStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumStatement(this);
}


antlrcpp::Any qasm3Parser::QuantumStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumStatementContext* qasm3Parser::quantumStatement() {
  QuantumStatementContext *_localctx = _tracker.createInstance<QuantumStatementContext>(_ctx, getState());
  enterRule(_localctx, 90, qasm3Parser::RuleQuantumStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(527);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__21:
      case qasm3Parser::T__22:
      case qasm3Parser::T__23:
      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26:
      case qasm3Parser::T__27:
      case qasm3Parser::T__28:
      case qasm3Parser::T__30:
      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(523);
        quantumInstruction();
        setState(524);
        match(qasm3Parser::SEMICOLON);
        break;
      }

      case qasm3Parser::T__89:
      case qasm3Parser::T__90:
      case qasm3Parser::T__92:
      case qasm3Parser::T__93: {
        enterOuterAlt(_localctx, 2);
        setState(526);
        timingStatement();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumInstructionContext ------------------------------------------------------------------

qasm3Parser::QuantumInstructionContext::QuantumInstructionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::QuantumGateCallContext* qasm3Parser::QuantumInstructionContext::quantumGateCall() {
  return getRuleContext<qasm3Parser::QuantumGateCallContext>(0);
}

qasm3Parser::QuantumPhaseContext* qasm3Parser::QuantumInstructionContext::quantumPhase() {
  return getRuleContext<qasm3Parser::QuantumPhaseContext>(0);
}

qasm3Parser::QuantumMeasurementContext* qasm3Parser::QuantumInstructionContext::quantumMeasurement() {
  return getRuleContext<qasm3Parser::QuantumMeasurementContext>(0);
}

qasm3Parser::QuantumBarrierContext* qasm3Parser::QuantumInstructionContext::quantumBarrier() {
  return getRuleContext<qasm3Parser::QuantumBarrierContext>(0);
}


size_t qasm3Parser::QuantumInstructionContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumInstruction;
}

void qasm3Parser::QuantumInstructionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumInstruction(this);
}

void qasm3Parser::QuantumInstructionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumInstruction(this);
}


antlrcpp::Any qasm3Parser::QuantumInstructionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumInstruction(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumInstructionContext* qasm3Parser::quantumInstruction() {
  QuantumInstructionContext *_localctx = _tracker.createInstance<QuantumInstructionContext>(_ctx, getState());
  enterRule(_localctx, 92, qasm3Parser::RuleQuantumInstruction);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(533);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__21:
      case qasm3Parser::T__22:
      case qasm3Parser::T__26:
      case qasm3Parser::T__27:
      case qasm3Parser::T__28:
      case qasm3Parser::T__30:
      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(529);
        quantumGateCall();
        break;
      }

      case qasm3Parser::T__23: {
        enterOuterAlt(_localctx, 2);
        setState(530);
        quantumPhase();
        break;
      }

      case qasm3Parser::T__24: {
        enterOuterAlt(_localctx, 3);
        setState(531);
        quantumMeasurement();
        break;
      }

      case qasm3Parser::T__25: {
        enterOuterAlt(_localctx, 4);
        setState(532);
        quantumBarrier();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumPhaseContext ------------------------------------------------------------------

qasm3Parser::QuantumPhaseContext::QuantumPhaseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::QuantumPhaseContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::QuantumPhaseContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

tree::TerminalNode* qasm3Parser::QuantumPhaseContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}


size_t qasm3Parser::QuantumPhaseContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumPhase;
}

void qasm3Parser::QuantumPhaseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumPhase(this);
}

void qasm3Parser::QuantumPhaseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumPhase(this);
}


antlrcpp::Any qasm3Parser::QuantumPhaseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumPhase(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumPhaseContext* qasm3Parser::quantumPhase() {
  QuantumPhaseContext *_localctx = _tracker.createInstance<QuantumPhaseContext>(_ctx, getState());
  enterRule(_localctx, 94, qasm3Parser::RuleQuantumPhase);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(535);
    match(qasm3Parser::T__23);
    setState(536);
    match(qasm3Parser::LPAREN);
    setState(537);
    match(qasm3Parser::Identifier);
    setState(538);
    match(qasm3Parser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumMeasurementContext ------------------------------------------------------------------

qasm3Parser::QuantumMeasurementContext::QuantumMeasurementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::QuantumMeasurementContext::indexIdentifierList() {
  return getRuleContext<qasm3Parser::IndexIdentifierListContext>(0);
}


size_t qasm3Parser::QuantumMeasurementContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumMeasurement;
}

void qasm3Parser::QuantumMeasurementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumMeasurement(this);
}

void qasm3Parser::QuantumMeasurementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumMeasurement(this);
}


antlrcpp::Any qasm3Parser::QuantumMeasurementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumMeasurement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumMeasurementContext* qasm3Parser::quantumMeasurement() {
  QuantumMeasurementContext *_localctx = _tracker.createInstance<QuantumMeasurementContext>(_ctx, getState());
  enterRule(_localctx, 96, qasm3Parser::RuleQuantumMeasurement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(540);
    match(qasm3Parser::T__24);
    setState(541);
    indexIdentifierList();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumMeasurementAssignmentContext ------------------------------------------------------------------

qasm3Parser::QuantumMeasurementAssignmentContext::QuantumMeasurementAssignmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::QuantumMeasurementContext* qasm3Parser::QuantumMeasurementAssignmentContext::quantumMeasurement() {
  return getRuleContext<qasm3Parser::QuantumMeasurementContext>(0);
}

tree::TerminalNode* qasm3Parser::QuantumMeasurementAssignmentContext::ARROW() {
  return getToken(qasm3Parser::ARROW, 0);
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::QuantumMeasurementAssignmentContext::indexIdentifierList() {
  return getRuleContext<qasm3Parser::IndexIdentifierListContext>(0);
}

tree::TerminalNode* qasm3Parser::QuantumMeasurementAssignmentContext::EQUALS() {
  return getToken(qasm3Parser::EQUALS, 0);
}


size_t qasm3Parser::QuantumMeasurementAssignmentContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumMeasurementAssignment;
}

void qasm3Parser::QuantumMeasurementAssignmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumMeasurementAssignment(this);
}

void qasm3Parser::QuantumMeasurementAssignmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumMeasurementAssignment(this);
}


antlrcpp::Any qasm3Parser::QuantumMeasurementAssignmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumMeasurementAssignment(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumMeasurementAssignmentContext* qasm3Parser::quantumMeasurementAssignment() {
  QuantumMeasurementAssignmentContext *_localctx = _tracker.createInstance<QuantumMeasurementAssignmentContext>(_ctx, getState());
  enterRule(_localctx, 98, qasm3Parser::RuleQuantumMeasurementAssignment);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(552);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__24: {
        enterOuterAlt(_localctx, 1);
        setState(543);
        quantumMeasurement();
        setState(546);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == qasm3Parser::ARROW) {
          setState(544);
          match(qasm3Parser::ARROW);
          setState(545);
          indexIdentifierList();
        }
        break;
      }

      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(548);
        indexIdentifierList();
        setState(549);
        match(qasm3Parser::EQUALS);
        setState(550);
        quantumMeasurement();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumBarrierContext ------------------------------------------------------------------

qasm3Parser::QuantumBarrierContext::QuantumBarrierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::QuantumBarrierContext::indexIdentifierList() {
  return getRuleContext<qasm3Parser::IndexIdentifierListContext>(0);
}


size_t qasm3Parser::QuantumBarrierContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumBarrier;
}

void qasm3Parser::QuantumBarrierContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumBarrier(this);
}

void qasm3Parser::QuantumBarrierContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumBarrier(this);
}


antlrcpp::Any qasm3Parser::QuantumBarrierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumBarrier(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumBarrierContext* qasm3Parser::quantumBarrier() {
  QuantumBarrierContext *_localctx = _tracker.createInstance<QuantumBarrierContext>(_ctx, getState());
  enterRule(_localctx, 100, qasm3Parser::RuleQuantumBarrier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(554);
    match(qasm3Parser::T__25);
    setState(555);
    indexIdentifierList();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumGateModifierContext ------------------------------------------------------------------

qasm3Parser::QuantumGateModifierContext::QuantumGateModifierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::QuantumGateModifierContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

qasm3Parser::ExpressionContext* qasm3Parser::QuantumGateModifierContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::QuantumGateModifierContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}


size_t qasm3Parser::QuantumGateModifierContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumGateModifier;
}

void qasm3Parser::QuantumGateModifierContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumGateModifier(this);
}

void qasm3Parser::QuantumGateModifierContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumGateModifier(this);
}


antlrcpp::Any qasm3Parser::QuantumGateModifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumGateModifier(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumGateModifierContext* qasm3Parser::quantumGateModifier() {
  QuantumGateModifierContext *_localctx = _tracker.createInstance<QuantumGateModifierContext>(_ctx, getState());
  enterRule(_localctx, 102, qasm3Parser::RuleQuantumGateModifier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(564);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__26: {
        setState(557);
        match(qasm3Parser::T__26);
        break;
      }

      case qasm3Parser::T__27: {
        setState(558);
        match(qasm3Parser::T__27);
        setState(559);
        match(qasm3Parser::LPAREN);
        setState(560);
        expression(0);
        setState(561);
        match(qasm3Parser::RPAREN);
        break;
      }

      case qasm3Parser::T__28: {
        setState(563);
        match(qasm3Parser::T__28);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(566);
    match(qasm3Parser::T__29);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumGateCallContext ------------------------------------------------------------------

qasm3Parser::QuantumGateCallContext::QuantumGateCallContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::QuantumGateNameContext* qasm3Parser::QuantumGateCallContext::quantumGateName() {
  return getRuleContext<qasm3Parser::QuantumGateNameContext>(0);
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::QuantumGateCallContext::indexIdentifierList() {
  return getRuleContext<qasm3Parser::IndexIdentifierListContext>(0);
}

tree::TerminalNode* qasm3Parser::QuantumGateCallContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::QuantumGateCallContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

qasm3Parser::ExpressionListContext* qasm3Parser::QuantumGateCallContext::expressionList() {
  return getRuleContext<qasm3Parser::ExpressionListContext>(0);
}


size_t qasm3Parser::QuantumGateCallContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumGateCall;
}

void qasm3Parser::QuantumGateCallContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumGateCall(this);
}

void qasm3Parser::QuantumGateCallContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumGateCall(this);
}


antlrcpp::Any qasm3Parser::QuantumGateCallContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumGateCall(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumGateCallContext* qasm3Parser::quantumGateCall() {
  QuantumGateCallContext *_localctx = _tracker.createInstance<QuantumGateCallContext>(_ctx, getState());
  enterRule(_localctx, 104, qasm3Parser::RuleQuantumGateCall);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(568);
    quantumGateName();
    setState(574);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(569);
      match(qasm3Parser::LPAREN);
      setState(571);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__15)
        | (1ULL << qasm3Parser::T__16)
        | (1ULL << qasm3Parser::T__31)
        | (1ULL << qasm3Parser::T__32)
        | (1ULL << qasm3Parser::T__51)
        | (1ULL << qasm3Parser::T__52)
        | (1ULL << qasm3Parser::T__53)
        | (1ULL << qasm3Parser::T__54)
        | (1ULL << qasm3Parser::T__55)
        | (1ULL << qasm3Parser::T__56)
        | (1ULL << qasm3Parser::T__57)
        | (1ULL << qasm3Parser::T__58)
        | (1ULL << qasm3Parser::T__59)
        | (1ULL << qasm3Parser::T__60)
        | (1ULL << qasm3Parser::T__61)
        | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
        | (1ULL << (qasm3Parser::T__87 - 64))
        | (1ULL << (qasm3Parser::T__88 - 64))
        | (1ULL << (qasm3Parser::T__91 - 64))
        | (1ULL << (qasm3Parser::LPAREN - 64))
        | (1ULL << (qasm3Parser::MINUS - 64))
        | (1ULL << (qasm3Parser::Constant - 64))
        | (1ULL << (qasm3Parser::Integer - 64))
        | (1ULL << (qasm3Parser::Identifier - 64))
        | (1ULL << (qasm3Parser::RealNumber - 64))
        | (1ULL << (qasm3Parser::TimingLiteral - 64))
        | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
        setState(570);
        expressionList();
      }
      setState(573);
      match(qasm3Parser::RPAREN);
    }
    setState(576);
    indexIdentifierList();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumGateNameContext ------------------------------------------------------------------

qasm3Parser::QuantumGateNameContext::QuantumGateNameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::QuantumGateNameContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::QuantumGateModifierContext* qasm3Parser::QuantumGateNameContext::quantumGateModifier() {
  return getRuleContext<qasm3Parser::QuantumGateModifierContext>(0);
}

qasm3Parser::QuantumGateNameContext* qasm3Parser::QuantumGateNameContext::quantumGateName() {
  return getRuleContext<qasm3Parser::QuantumGateNameContext>(0);
}


size_t qasm3Parser::QuantumGateNameContext::getRuleIndex() const {
  return qasm3Parser::RuleQuantumGateName;
}

void qasm3Parser::QuantumGateNameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumGateName(this);
}

void qasm3Parser::QuantumGateNameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumGateName(this);
}


antlrcpp::Any qasm3Parser::QuantumGateNameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitQuantumGateName(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::QuantumGateNameContext* qasm3Parser::quantumGateName() {
  QuantumGateNameContext *_localctx = _tracker.createInstance<QuantumGateNameContext>(_ctx, getState());
  enterRule(_localctx, 106, qasm3Parser::RuleQuantumGateName);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(585);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__21: {
        enterOuterAlt(_localctx, 1);
        setState(578);
        match(qasm3Parser::T__21);
        break;
      }

      case qasm3Parser::T__22: {
        enterOuterAlt(_localctx, 2);
        setState(579);
        match(qasm3Parser::T__22);
        break;
      }

      case qasm3Parser::T__30: {
        enterOuterAlt(_localctx, 3);
        setState(580);
        match(qasm3Parser::T__30);
        break;
      }

      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 4);
        setState(581);
        match(qasm3Parser::Identifier);
        break;
      }

      case qasm3Parser::T__26:
      case qasm3Parser::T__27:
      case qasm3Parser::T__28: {
        enterOuterAlt(_localctx, 5);
        setState(582);
        quantumGateModifier();
        setState(583);
        quantumGateName();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryOperatorContext ------------------------------------------------------------------

qasm3Parser::UnaryOperatorContext::UnaryOperatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::UnaryOperatorContext::getRuleIndex() const {
  return qasm3Parser::RuleUnaryOperator;
}

void qasm3Parser::UnaryOperatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryOperator(this);
}

void qasm3Parser::UnaryOperatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryOperator(this);
}


antlrcpp::Any qasm3Parser::UnaryOperatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitUnaryOperator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::UnaryOperatorContext* qasm3Parser::unaryOperator() {
  UnaryOperatorContext *_localctx = _tracker.createInstance<UnaryOperatorContext>(_ctx, getState());
  enterRule(_localctx, 108, qasm3Parser::RuleUnaryOperator);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(587);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__31

    || _la == qasm3Parser::T__32)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RelationalOperatorContext ------------------------------------------------------------------

qasm3Parser::RelationalOperatorContext::RelationalOperatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::RelationalOperatorContext::getRuleIndex() const {
  return qasm3Parser::RuleRelationalOperator;
}

void qasm3Parser::RelationalOperatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRelationalOperator(this);
}

void qasm3Parser::RelationalOperatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRelationalOperator(this);
}


antlrcpp::Any qasm3Parser::RelationalOperatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitRelationalOperator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::RelationalOperatorContext* qasm3Parser::relationalOperator() {
  RelationalOperatorContext *_localctx = _tracker.createInstance<RelationalOperatorContext>(_ctx, getState());
  enterRule(_localctx, 110, qasm3Parser::RuleRelationalOperator);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(589);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__33)
      | (1ULL << qasm3Parser::T__34)
      | (1ULL << qasm3Parser::T__35)
      | (1ULL << qasm3Parser::T__36)
      | (1ULL << qasm3Parser::T__37)
      | (1ULL << qasm3Parser::T__38))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalOperatorContext ------------------------------------------------------------------

qasm3Parser::LogicalOperatorContext::LogicalOperatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::LogicalOperatorContext::getRuleIndex() const {
  return qasm3Parser::RuleLogicalOperator;
}

void qasm3Parser::LogicalOperatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogicalOperator(this);
}

void qasm3Parser::LogicalOperatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogicalOperator(this);
}


antlrcpp::Any qasm3Parser::LogicalOperatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitLogicalOperator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::LogicalOperatorContext* qasm3Parser::logicalOperator() {
  LogicalOperatorContext *_localctx = _tracker.createInstance<LogicalOperatorContext>(_ctx, getState());
  enterRule(_localctx, 112, qasm3Parser::RuleLogicalOperator);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(591);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__19

    || _la == qasm3Parser::T__39)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionStatementContext ------------------------------------------------------------------

qasm3Parser::ExpressionStatementContext::ExpressionStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ExpressionContext* qasm3Parser::ExpressionStatementContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::ExpressionStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::ExpressionStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleExpressionStatement;
}

void qasm3Parser::ExpressionStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionStatement(this);
}

void qasm3Parser::ExpressionStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionStatement(this);
}


antlrcpp::Any qasm3Parser::ExpressionStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitExpressionStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ExpressionStatementContext* qasm3Parser::expressionStatement() {
  ExpressionStatementContext *_localctx = _tracker.createInstance<ExpressionStatementContext>(_ctx, getState());
  enterRule(_localctx, 114, qasm3Parser::RuleExpressionStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(593);
    expression(0);
    setState(594);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

qasm3Parser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ExpressionTerminatorContext* qasm3Parser::ExpressionContext::expressionTerminator() {
  return getRuleContext<qasm3Parser::ExpressionTerminatorContext>(0);
}

qasm3Parser::UnaryExpressionContext* qasm3Parser::ExpressionContext::unaryExpression() {
  return getRuleContext<qasm3Parser::UnaryExpressionContext>(0);
}

qasm3Parser::XOrExpressionContext* qasm3Parser::ExpressionContext::xOrExpression() {
  return getRuleContext<qasm3Parser::XOrExpressionContext>(0);
}

qasm3Parser::ExpressionContext* qasm3Parser::ExpressionContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}


size_t qasm3Parser::ExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleExpression;
}

void qasm3Parser::ExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression(this);
}

void qasm3Parser::ExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression(this);
}


antlrcpp::Any qasm3Parser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::ExpressionContext* qasm3Parser::expression() {
   return expression(0);
}

qasm3Parser::ExpressionContext* qasm3Parser::expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, parentState);
  qasm3Parser::ExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 116;
  enterRecursionRule(_localctx, 116, qasm3Parser::RuleExpression, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(600);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 47, _ctx)) {
    case 1: {
      setState(597);
      expressionTerminator(0);
      break;
    }

    case 2: {
      setState(598);
      unaryExpression();
      break;
    }

    case 3: {
      setState(599);
      xOrExpression(0);
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(607);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 48, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleExpression);
        setState(602);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(603);
        match(qasm3Parser::T__40);
        setState(604);
        xOrExpression(0); 
      }
      setState(609);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 48, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- XOrExpressionContext ------------------------------------------------------------------

qasm3Parser::XOrExpressionContext::XOrExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::BitAndExpressionContext* qasm3Parser::XOrExpressionContext::bitAndExpression() {
  return getRuleContext<qasm3Parser::BitAndExpressionContext>(0);
}

qasm3Parser::XOrExpressionContext* qasm3Parser::XOrExpressionContext::xOrExpression() {
  return getRuleContext<qasm3Parser::XOrExpressionContext>(0);
}


size_t qasm3Parser::XOrExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleXOrExpression;
}

void qasm3Parser::XOrExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterXOrExpression(this);
}

void qasm3Parser::XOrExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitXOrExpression(this);
}


antlrcpp::Any qasm3Parser::XOrExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitXOrExpression(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::XOrExpressionContext* qasm3Parser::xOrExpression() {
   return xOrExpression(0);
}

qasm3Parser::XOrExpressionContext* qasm3Parser::xOrExpression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::XOrExpressionContext *_localctx = _tracker.createInstance<XOrExpressionContext>(_ctx, parentState);
  qasm3Parser::XOrExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 118;
  enterRecursionRule(_localctx, 118, qasm3Parser::RuleXOrExpression, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(611);
    bitAndExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(618);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 49, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<XOrExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleXOrExpression);
        setState(613);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(614);
        match(qasm3Parser::T__41);
        setState(615);
        bitAndExpression(0); 
      }
      setState(620);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 49, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- BitAndExpressionContext ------------------------------------------------------------------

qasm3Parser::BitAndExpressionContext::BitAndExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::BitShiftExpressionContext* qasm3Parser::BitAndExpressionContext::bitShiftExpression() {
  return getRuleContext<qasm3Parser::BitShiftExpressionContext>(0);
}

qasm3Parser::BitAndExpressionContext* qasm3Parser::BitAndExpressionContext::bitAndExpression() {
  return getRuleContext<qasm3Parser::BitAndExpressionContext>(0);
}


size_t qasm3Parser::BitAndExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleBitAndExpression;
}

void qasm3Parser::BitAndExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBitAndExpression(this);
}

void qasm3Parser::BitAndExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBitAndExpression(this);
}


antlrcpp::Any qasm3Parser::BitAndExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBitAndExpression(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::BitAndExpressionContext* qasm3Parser::bitAndExpression() {
   return bitAndExpression(0);
}

qasm3Parser::BitAndExpressionContext* qasm3Parser::bitAndExpression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::BitAndExpressionContext *_localctx = _tracker.createInstance<BitAndExpressionContext>(_ctx, parentState);
  qasm3Parser::BitAndExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 120;
  enterRecursionRule(_localctx, 120, qasm3Parser::RuleBitAndExpression, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(622);
    bitShiftExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(629);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 50, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<BitAndExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleBitAndExpression);
        setState(624);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(625);
        match(qasm3Parser::T__42);
        setState(626);
        bitShiftExpression(0); 
      }
      setState(631);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 50, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- BitShiftExpressionContext ------------------------------------------------------------------

qasm3Parser::BitShiftExpressionContext::BitShiftExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::AdditiveExpressionContext* qasm3Parser::BitShiftExpressionContext::additiveExpression() {
  return getRuleContext<qasm3Parser::AdditiveExpressionContext>(0);
}

qasm3Parser::BitShiftExpressionContext* qasm3Parser::BitShiftExpressionContext::bitShiftExpression() {
  return getRuleContext<qasm3Parser::BitShiftExpressionContext>(0);
}


size_t qasm3Parser::BitShiftExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleBitShiftExpression;
}

void qasm3Parser::BitShiftExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBitShiftExpression(this);
}

void qasm3Parser::BitShiftExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBitShiftExpression(this);
}


antlrcpp::Any qasm3Parser::BitShiftExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBitShiftExpression(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::BitShiftExpressionContext* qasm3Parser::bitShiftExpression() {
   return bitShiftExpression(0);
}

qasm3Parser::BitShiftExpressionContext* qasm3Parser::bitShiftExpression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::BitShiftExpressionContext *_localctx = _tracker.createInstance<BitShiftExpressionContext>(_ctx, parentState);
  qasm3Parser::BitShiftExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 122;
  enterRecursionRule(_localctx, 122, qasm3Parser::RuleBitShiftExpression, precedence);

    size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(633);
    additiveExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(640);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 51, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<BitShiftExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleBitShiftExpression);
        setState(635);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(636);
        _la = _input->LA(1);
        if (!(_la == qasm3Parser::T__43

        || _la == qasm3Parser::T__44)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(637);
        additiveExpression(0); 
      }
      setState(642);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 51, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- AdditiveExpressionContext ------------------------------------------------------------------

qasm3Parser::AdditiveExpressionContext::AdditiveExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::MultiplicativeExpressionContext* qasm3Parser::AdditiveExpressionContext::multiplicativeExpression() {
  return getRuleContext<qasm3Parser::MultiplicativeExpressionContext>(0);
}

qasm3Parser::AdditiveExpressionContext* qasm3Parser::AdditiveExpressionContext::additiveExpression() {
  return getRuleContext<qasm3Parser::AdditiveExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::AdditiveExpressionContext::MINUS() {
  return getToken(qasm3Parser::MINUS, 0);
}


size_t qasm3Parser::AdditiveExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleAdditiveExpression;
}

void qasm3Parser::AdditiveExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAdditiveExpression(this);
}

void qasm3Parser::AdditiveExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAdditiveExpression(this);
}


antlrcpp::Any qasm3Parser::AdditiveExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitAdditiveExpression(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::AdditiveExpressionContext* qasm3Parser::additiveExpression() {
   return additiveExpression(0);
}

qasm3Parser::AdditiveExpressionContext* qasm3Parser::additiveExpression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::AdditiveExpressionContext *_localctx = _tracker.createInstance<AdditiveExpressionContext>(_ctx, parentState);
  qasm3Parser::AdditiveExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 124;
  enterRecursionRule(_localctx, 124, qasm3Parser::RuleAdditiveExpression, precedence);

    size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(644);
    multiplicativeExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(651);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<AdditiveExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleAdditiveExpression);
        setState(646);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(647);
        dynamic_cast<AdditiveExpressionContext *>(_localctx)->binary_op = _input->LT(1);
        _la = _input->LA(1);
        if (!(_la == qasm3Parser::T__45 || _la == qasm3Parser::MINUS)) {
          dynamic_cast<AdditiveExpressionContext *>(_localctx)->binary_op = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(648);
        multiplicativeExpression(0); 
      }
      setState(653);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- MultiplicativeExpressionContext ------------------------------------------------------------------

qasm3Parser::MultiplicativeExpressionContext::MultiplicativeExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ExpressionTerminatorContext* qasm3Parser::MultiplicativeExpressionContext::expressionTerminator() {
  return getRuleContext<qasm3Parser::ExpressionTerminatorContext>(0);
}

qasm3Parser::UnaryExpressionContext* qasm3Parser::MultiplicativeExpressionContext::unaryExpression() {
  return getRuleContext<qasm3Parser::UnaryExpressionContext>(0);
}

qasm3Parser::MultiplicativeExpressionContext* qasm3Parser::MultiplicativeExpressionContext::multiplicativeExpression() {
  return getRuleContext<qasm3Parser::MultiplicativeExpressionContext>(0);
}


size_t qasm3Parser::MultiplicativeExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleMultiplicativeExpression;
}

void qasm3Parser::MultiplicativeExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiplicativeExpression(this);
}

void qasm3Parser::MultiplicativeExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiplicativeExpression(this);
}


antlrcpp::Any qasm3Parser::MultiplicativeExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitMultiplicativeExpression(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::MultiplicativeExpressionContext* qasm3Parser::multiplicativeExpression() {
   return multiplicativeExpression(0);
}

qasm3Parser::MultiplicativeExpressionContext* qasm3Parser::multiplicativeExpression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::MultiplicativeExpressionContext *_localctx = _tracker.createInstance<MultiplicativeExpressionContext>(_ctx, parentState);
  qasm3Parser::MultiplicativeExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 126;
  enterRecursionRule(_localctx, 126, qasm3Parser::RuleMultiplicativeExpression, precedence);

    size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(657);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__7:
      case qasm3Parser::T__8:
      case qasm3Parser::T__9:
      case qasm3Parser::T__10:
      case qasm3Parser::T__11:
      case qasm3Parser::T__12:
      case qasm3Parser::T__13:
      case qasm3Parser::T__14:
      case qasm3Parser::T__15:
      case qasm3Parser::T__16:
      case qasm3Parser::T__51:
      case qasm3Parser::T__52:
      case qasm3Parser::T__53:
      case qasm3Parser::T__54:
      case qasm3Parser::T__55:
      case qasm3Parser::T__56:
      case qasm3Parser::T__57:
      case qasm3Parser::T__58:
      case qasm3Parser::T__59:
      case qasm3Parser::T__60:
      case qasm3Parser::T__61:
      case qasm3Parser::T__62:
      case qasm3Parser::T__63:
      case qasm3Parser::T__87:
      case qasm3Parser::T__88:
      case qasm3Parser::T__91:
      case qasm3Parser::LPAREN:
      case qasm3Parser::MINUS:
      case qasm3Parser::Constant:
      case qasm3Parser::Integer:
      case qasm3Parser::Identifier:
      case qasm3Parser::RealNumber:
      case qasm3Parser::TimingLiteral:
      case qasm3Parser::StringLiteral: {
        setState(655);
        expressionTerminator(0);
        break;
      }

      case qasm3Parser::T__31:
      case qasm3Parser::T__32: {
        setState(656);
        unaryExpression();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(667);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 55, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<MultiplicativeExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleMultiplicativeExpression);
        setState(659);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(660);
        dynamic_cast<MultiplicativeExpressionContext *>(_localctx)->binary_op = _input->LT(1);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << qasm3Parser::T__46)
          | (1ULL << qasm3Parser::T__47)
          | (1ULL << qasm3Parser::T__48))) != 0))) {
          dynamic_cast<MultiplicativeExpressionContext *>(_localctx)->binary_op = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(663);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case qasm3Parser::T__7:
          case qasm3Parser::T__8:
          case qasm3Parser::T__9:
          case qasm3Parser::T__10:
          case qasm3Parser::T__11:
          case qasm3Parser::T__12:
          case qasm3Parser::T__13:
          case qasm3Parser::T__14:
          case qasm3Parser::T__15:
          case qasm3Parser::T__16:
          case qasm3Parser::T__51:
          case qasm3Parser::T__52:
          case qasm3Parser::T__53:
          case qasm3Parser::T__54:
          case qasm3Parser::T__55:
          case qasm3Parser::T__56:
          case qasm3Parser::T__57:
          case qasm3Parser::T__58:
          case qasm3Parser::T__59:
          case qasm3Parser::T__60:
          case qasm3Parser::T__61:
          case qasm3Parser::T__62:
          case qasm3Parser::T__63:
          case qasm3Parser::T__87:
          case qasm3Parser::T__88:
          case qasm3Parser::T__91:
          case qasm3Parser::LPAREN:
          case qasm3Parser::MINUS:
          case qasm3Parser::Constant:
          case qasm3Parser::Integer:
          case qasm3Parser::Identifier:
          case qasm3Parser::RealNumber:
          case qasm3Parser::TimingLiteral:
          case qasm3Parser::StringLiteral: {
            setState(661);
            expressionTerminator(0);
            break;
          }

          case qasm3Parser::T__31:
          case qasm3Parser::T__32: {
            setState(662);
            unaryExpression();
            break;
          }

        default:
          throw NoViableAltException(this);
        } 
      }
      setState(669);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 55, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- UnaryExpressionContext ------------------------------------------------------------------

qasm3Parser::UnaryExpressionContext::UnaryExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::UnaryOperatorContext* qasm3Parser::UnaryExpressionContext::unaryOperator() {
  return getRuleContext<qasm3Parser::UnaryOperatorContext>(0);
}

qasm3Parser::ExpressionTerminatorContext* qasm3Parser::UnaryExpressionContext::expressionTerminator() {
  return getRuleContext<qasm3Parser::ExpressionTerminatorContext>(0);
}


size_t qasm3Parser::UnaryExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleUnaryExpression;
}

void qasm3Parser::UnaryExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryExpression(this);
}

void qasm3Parser::UnaryExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryExpression(this);
}


antlrcpp::Any qasm3Parser::UnaryExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitUnaryExpression(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::UnaryExpressionContext* qasm3Parser::unaryExpression() {
  UnaryExpressionContext *_localctx = _tracker.createInstance<UnaryExpressionContext>(_ctx, getState());
  enterRule(_localctx, 128, qasm3Parser::RuleUnaryExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(670);
    unaryOperator();
    setState(671);
    expressionTerminator(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionTerminatorContext ------------------------------------------------------------------

qasm3Parser::ExpressionTerminatorContext::ExpressionTerminatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::Constant() {
  return getToken(qasm3Parser::Constant, 0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::Integer() {
  return getToken(qasm3Parser::Integer, 0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::RealNumber() {
  return getToken(qasm3Parser::RealNumber, 0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::StringLiteral() {
  return getToken(qasm3Parser::StringLiteral, 0);
}

qasm3Parser::BuiltInCallContext* qasm3Parser::ExpressionTerminatorContext::builtInCall() {
  return getRuleContext<qasm3Parser::BuiltInCallContext>(0);
}

qasm3Parser::KernelCallContext* qasm3Parser::ExpressionTerminatorContext::kernelCall() {
  return getRuleContext<qasm3Parser::KernelCallContext>(0);
}

qasm3Parser::SubroutineCallContext* qasm3Parser::ExpressionTerminatorContext::subroutineCall() {
  return getRuleContext<qasm3Parser::SubroutineCallContext>(0);
}

qasm3Parser::TimingTerminatorContext* qasm3Parser::ExpressionTerminatorContext::timingTerminator() {
  return getRuleContext<qasm3Parser::TimingTerminatorContext>(0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::MINUS() {
  return getToken(qasm3Parser::MINUS, 0);
}

qasm3Parser::ExpressionTerminatorContext* qasm3Parser::ExpressionTerminatorContext::expressionTerminator() {
  return getRuleContext<qasm3Parser::ExpressionTerminatorContext>(0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

qasm3Parser::ExpressionContext* qasm3Parser::ExpressionTerminatorContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::LBRACKET() {
  return getToken(qasm3Parser::LBRACKET, 0);
}

tree::TerminalNode* qasm3Parser::ExpressionTerminatorContext::RBRACKET() {
  return getToken(qasm3Parser::RBRACKET, 0);
}

qasm3Parser::IncrementorContext* qasm3Parser::ExpressionTerminatorContext::incrementor() {
  return getRuleContext<qasm3Parser::IncrementorContext>(0);
}


size_t qasm3Parser::ExpressionTerminatorContext::getRuleIndex() const {
  return qasm3Parser::RuleExpressionTerminator;
}

void qasm3Parser::ExpressionTerminatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionTerminator(this);
}

void qasm3Parser::ExpressionTerminatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionTerminator(this);
}


antlrcpp::Any qasm3Parser::ExpressionTerminatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitExpressionTerminator(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::ExpressionTerminatorContext* qasm3Parser::expressionTerminator() {
   return expressionTerminator(0);
}

qasm3Parser::ExpressionTerminatorContext* qasm3Parser::expressionTerminator(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::ExpressionTerminatorContext *_localctx = _tracker.createInstance<ExpressionTerminatorContext>(_ctx, parentState);
  qasm3Parser::ExpressionTerminatorContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 130;
  enterRecursionRule(_localctx, 130, qasm3Parser::RuleExpressionTerminator, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(689);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 56, _ctx)) {
    case 1: {
      setState(674);
      match(qasm3Parser::Constant);
      break;
    }

    case 2: {
      setState(675);
      match(qasm3Parser::Integer);
      break;
    }

    case 3: {
      setState(676);
      match(qasm3Parser::RealNumber);
      break;
    }

    case 4: {
      setState(677);
      match(qasm3Parser::Identifier);
      break;
    }

    case 5: {
      setState(678);
      match(qasm3Parser::StringLiteral);
      break;
    }

    case 6: {
      setState(679);
      builtInCall();
      break;
    }

    case 7: {
      setState(680);
      kernelCall();
      break;
    }

    case 8: {
      setState(681);
      subroutineCall();
      break;
    }

    case 9: {
      setState(682);
      timingTerminator();
      break;
    }

    case 10: {
      setState(683);
      match(qasm3Parser::MINUS);
      setState(684);
      expressionTerminator(4);
      break;
    }

    case 11: {
      setState(685);
      match(qasm3Parser::LPAREN);
      setState(686);
      expression(0);
      setState(687);
      match(qasm3Parser::RPAREN);
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(700);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 58, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(698);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 57, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExpressionTerminatorContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpressionTerminator);
          setState(691);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(692);
          match(qasm3Parser::LBRACKET);
          setState(693);
          expression(0);
          setState(694);
          match(qasm3Parser::RBRACKET);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExpressionTerminatorContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpressionTerminator);
          setState(696);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(697);
          incrementor();
          break;
        }

        default:
          break;
        } 
      }
      setState(702);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 58, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- IncrementorContext ------------------------------------------------------------------

qasm3Parser::IncrementorContext::IncrementorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::IncrementorContext::getRuleIndex() const {
  return qasm3Parser::RuleIncrementor;
}

void qasm3Parser::IncrementorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIncrementor(this);
}

void qasm3Parser::IncrementorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIncrementor(this);
}


antlrcpp::Any qasm3Parser::IncrementorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitIncrementor(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::IncrementorContext* qasm3Parser::incrementor() {
  IncrementorContext *_localctx = _tracker.createInstance<IncrementorContext>(_ctx, getState());
  enterRule(_localctx, 132, qasm3Parser::RuleIncrementor);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(703);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__49

    || _la == qasm3Parser::T__50)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BuiltInCallContext ------------------------------------------------------------------

qasm3Parser::BuiltInCallContext::BuiltInCallContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::BuiltInCallContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

qasm3Parser::ExpressionListContext* qasm3Parser::BuiltInCallContext::expressionList() {
  return getRuleContext<qasm3Parser::ExpressionListContext>(0);
}

tree::TerminalNode* qasm3Parser::BuiltInCallContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

qasm3Parser::BuiltInMathContext* qasm3Parser::BuiltInCallContext::builtInMath() {
  return getRuleContext<qasm3Parser::BuiltInMathContext>(0);
}

qasm3Parser::CastOperatorContext* qasm3Parser::BuiltInCallContext::castOperator() {
  return getRuleContext<qasm3Parser::CastOperatorContext>(0);
}


size_t qasm3Parser::BuiltInCallContext::getRuleIndex() const {
  return qasm3Parser::RuleBuiltInCall;
}

void qasm3Parser::BuiltInCallContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBuiltInCall(this);
}

void qasm3Parser::BuiltInCallContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBuiltInCall(this);
}


antlrcpp::Any qasm3Parser::BuiltInCallContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBuiltInCall(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::BuiltInCallContext* qasm3Parser::builtInCall() {
  BuiltInCallContext *_localctx = _tracker.createInstance<BuiltInCallContext>(_ctx, getState());
  enterRule(_localctx, 134, qasm3Parser::RuleBuiltInCall);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(707);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__51:
      case qasm3Parser::T__52:
      case qasm3Parser::T__53:
      case qasm3Parser::T__54:
      case qasm3Parser::T__55:
      case qasm3Parser::T__56:
      case qasm3Parser::T__57:
      case qasm3Parser::T__58:
      case qasm3Parser::T__59:
      case qasm3Parser::T__60:
      case qasm3Parser::T__61:
      case qasm3Parser::T__62:
      case qasm3Parser::T__63: {
        setState(705);
        builtInMath();
        break;
      }

      case qasm3Parser::T__7:
      case qasm3Parser::T__8:
      case qasm3Parser::T__9:
      case qasm3Parser::T__10:
      case qasm3Parser::T__11:
      case qasm3Parser::T__12:
      case qasm3Parser::T__13:
      case qasm3Parser::T__14:
      case qasm3Parser::T__15:
      case qasm3Parser::T__16:
      case qasm3Parser::T__87:
      case qasm3Parser::T__88: {
        setState(706);
        castOperator();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(709);
    match(qasm3Parser::LPAREN);
    setState(710);
    expressionList();
    setState(711);
    match(qasm3Parser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BuiltInMathContext ------------------------------------------------------------------

qasm3Parser::BuiltInMathContext::BuiltInMathContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::BuiltInMathContext::getRuleIndex() const {
  return qasm3Parser::RuleBuiltInMath;
}

void qasm3Parser::BuiltInMathContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBuiltInMath(this);
}

void qasm3Parser::BuiltInMathContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBuiltInMath(this);
}


antlrcpp::Any qasm3Parser::BuiltInMathContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBuiltInMath(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::BuiltInMathContext* qasm3Parser::builtInMath() {
  BuiltInMathContext *_localctx = _tracker.createInstance<BuiltInMathContext>(_ctx, getState());
  enterRule(_localctx, 136, qasm3Parser::RuleBuiltInMath);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(713);
    _la = _input->LA(1);
    if (!(((((_la - 52) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 52)) & ((1ULL << (qasm3Parser::T__51 - 52))
      | (1ULL << (qasm3Parser::T__52 - 52))
      | (1ULL << (qasm3Parser::T__53 - 52))
      | (1ULL << (qasm3Parser::T__54 - 52))
      | (1ULL << (qasm3Parser::T__55 - 52))
      | (1ULL << (qasm3Parser::T__56 - 52))
      | (1ULL << (qasm3Parser::T__57 - 52))
      | (1ULL << (qasm3Parser::T__58 - 52))
      | (1ULL << (qasm3Parser::T__59 - 52))
      | (1ULL << (qasm3Parser::T__60 - 52))
      | (1ULL << (qasm3Parser::T__61 - 52))
      | (1ULL << (qasm3Parser::T__62 - 52))
      | (1ULL << (qasm3Parser::T__63 - 52)))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CastOperatorContext ------------------------------------------------------------------

qasm3Parser::CastOperatorContext::CastOperatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ClassicalTypeContext* qasm3Parser::CastOperatorContext::classicalType() {
  return getRuleContext<qasm3Parser::ClassicalTypeContext>(0);
}


size_t qasm3Parser::CastOperatorContext::getRuleIndex() const {
  return qasm3Parser::RuleCastOperator;
}

void qasm3Parser::CastOperatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCastOperator(this);
}

void qasm3Parser::CastOperatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCastOperator(this);
}


antlrcpp::Any qasm3Parser::CastOperatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCastOperator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::CastOperatorContext* qasm3Parser::castOperator() {
  CastOperatorContext *_localctx = _tracker.createInstance<CastOperatorContext>(_ctx, getState());
  enterRule(_localctx, 138, qasm3Parser::RuleCastOperator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(715);
    classicalType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionListContext ------------------------------------------------------------------

qasm3Parser::ExpressionListContext::ExpressionListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::ExpressionContext *> qasm3Parser::ExpressionListContext::expression() {
  return getRuleContexts<qasm3Parser::ExpressionContext>();
}

qasm3Parser::ExpressionContext* qasm3Parser::ExpressionListContext::expression(size_t i) {
  return getRuleContext<qasm3Parser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> qasm3Parser::ExpressionListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::ExpressionListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::ExpressionListContext::getRuleIndex() const {
  return qasm3Parser::RuleExpressionList;
}

void qasm3Parser::ExpressionListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionList(this);
}

void qasm3Parser::ExpressionListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionList(this);
}


antlrcpp::Any qasm3Parser::ExpressionListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitExpressionList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ExpressionListContext* qasm3Parser::expressionList() {
  ExpressionListContext *_localctx = _tracker.createInstance<ExpressionListContext>(_ctx, getState());
  enterRule(_localctx, 140, qasm3Parser::RuleExpressionList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(722);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 60, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(717);
        expression(0);
        setState(718);
        match(qasm3Parser::COMMA); 
      }
      setState(724);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 60, _ctx);
    }
    setState(725);
    expression(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BooleanExpressionContext ------------------------------------------------------------------

qasm3Parser::BooleanExpressionContext::BooleanExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::MembershipTestContext* qasm3Parser::BooleanExpressionContext::membershipTest() {
  return getRuleContext<qasm3Parser::MembershipTestContext>(0);
}

qasm3Parser::ComparsionExpressionContext* qasm3Parser::BooleanExpressionContext::comparsionExpression() {
  return getRuleContext<qasm3Parser::ComparsionExpressionContext>(0);
}

qasm3Parser::BooleanExpressionContext* qasm3Parser::BooleanExpressionContext::booleanExpression() {
  return getRuleContext<qasm3Parser::BooleanExpressionContext>(0);
}

qasm3Parser::LogicalOperatorContext* qasm3Parser::BooleanExpressionContext::logicalOperator() {
  return getRuleContext<qasm3Parser::LogicalOperatorContext>(0);
}


size_t qasm3Parser::BooleanExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleBooleanExpression;
}

void qasm3Parser::BooleanExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBooleanExpression(this);
}

void qasm3Parser::BooleanExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBooleanExpression(this);
}


antlrcpp::Any qasm3Parser::BooleanExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBooleanExpression(this);
  else
    return visitor->visitChildren(this);
}


qasm3Parser::BooleanExpressionContext* qasm3Parser::booleanExpression() {
   return booleanExpression(0);
}

qasm3Parser::BooleanExpressionContext* qasm3Parser::booleanExpression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasm3Parser::BooleanExpressionContext *_localctx = _tracker.createInstance<BooleanExpressionContext>(_ctx, parentState);
  qasm3Parser::BooleanExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 142;
  enterRecursionRule(_localctx, 142, qasm3Parser::RuleBooleanExpression, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(730);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 61, _ctx)) {
    case 1: {
      setState(728);
      membershipTest();
      break;
    }

    case 2: {
      setState(729);
      comparsionExpression();
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(738);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 62, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<BooleanExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleBooleanExpression);
        setState(732);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(733);
        logicalOperator();
        setState(734);
        comparsionExpression(); 
      }
      setState(740);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 62, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ComparsionExpressionContext ------------------------------------------------------------------

qasm3Parser::ComparsionExpressionContext::ComparsionExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::ExpressionContext *> qasm3Parser::ComparsionExpressionContext::expression() {
  return getRuleContexts<qasm3Parser::ExpressionContext>();
}

qasm3Parser::ExpressionContext* qasm3Parser::ComparsionExpressionContext::expression(size_t i) {
  return getRuleContext<qasm3Parser::ExpressionContext>(i);
}

qasm3Parser::RelationalOperatorContext* qasm3Parser::ComparsionExpressionContext::relationalOperator() {
  return getRuleContext<qasm3Parser::RelationalOperatorContext>(0);
}


size_t qasm3Parser::ComparsionExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleComparsionExpression;
}

void qasm3Parser::ComparsionExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComparsionExpression(this);
}

void qasm3Parser::ComparsionExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComparsionExpression(this);
}


antlrcpp::Any qasm3Parser::ComparsionExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitComparsionExpression(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ComparsionExpressionContext* qasm3Parser::comparsionExpression() {
  ComparsionExpressionContext *_localctx = _tracker.createInstance<ComparsionExpressionContext>(_ctx, getState());
  enterRule(_localctx, 144, qasm3Parser::RuleComparsionExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(746);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 63, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(741);
      expression(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(742);
      expression(0);
      setState(743);
      relationalOperator();
      setState(744);
      expression(0);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EqualsExpressionContext ------------------------------------------------------------------

qasm3Parser::EqualsExpressionContext::EqualsExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::EqualsExpressionContext::EQUALS() {
  return getToken(qasm3Parser::EQUALS, 0);
}

qasm3Parser::ExpressionContext* qasm3Parser::EqualsExpressionContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}


size_t qasm3Parser::EqualsExpressionContext::getRuleIndex() const {
  return qasm3Parser::RuleEqualsExpression;
}

void qasm3Parser::EqualsExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEqualsExpression(this);
}

void qasm3Parser::EqualsExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEqualsExpression(this);
}


antlrcpp::Any qasm3Parser::EqualsExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitEqualsExpression(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::EqualsExpressionContext* qasm3Parser::equalsExpression() {
  EqualsExpressionContext *_localctx = _tracker.createInstance<EqualsExpressionContext>(_ctx, getState());
  enterRule(_localctx, 146, qasm3Parser::RuleEqualsExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(748);
    match(qasm3Parser::EQUALS);
    setState(749);
    expression(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentOperatorContext ------------------------------------------------------------------

qasm3Parser::AssignmentOperatorContext::AssignmentOperatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::AssignmentOperatorContext::EQUALS() {
  return getToken(qasm3Parser::EQUALS, 0);
}


size_t qasm3Parser::AssignmentOperatorContext::getRuleIndex() const {
  return qasm3Parser::RuleAssignmentOperator;
}

void qasm3Parser::AssignmentOperatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignmentOperator(this);
}

void qasm3Parser::AssignmentOperatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignmentOperator(this);
}


antlrcpp::Any qasm3Parser::AssignmentOperatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitAssignmentOperator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::AssignmentOperatorContext* qasm3Parser::assignmentOperator() {
  AssignmentOperatorContext *_localctx = _tracker.createInstance<AssignmentOperatorContext>(_ctx, getState());
  enterRule(_localctx, 148, qasm3Parser::RuleAssignmentOperator);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(751);
    _la = _input->LA(1);
    if (!(((((_la - 65) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 65)) & ((1ULL << (qasm3Parser::T__64 - 65))
      | (1ULL << (qasm3Parser::T__65 - 65))
      | (1ULL << (qasm3Parser::T__66 - 65))
      | (1ULL << (qasm3Parser::T__67 - 65))
      | (1ULL << (qasm3Parser::T__68 - 65))
      | (1ULL << (qasm3Parser::T__69 - 65))
      | (1ULL << (qasm3Parser::T__70 - 65))
      | (1ULL << (qasm3Parser::T__71 - 65))
      | (1ULL << (qasm3Parser::T__72 - 65))
      | (1ULL << (qasm3Parser::T__73 - 65))
      | (1ULL << (qasm3Parser::EQUALS - 65)))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EqualsAssignmentListContext ------------------------------------------------------------------

qasm3Parser::EqualsAssignmentListContext::EqualsAssignmentListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> qasm3Parser::EqualsAssignmentListContext::Identifier() {
  return getTokens(qasm3Parser::Identifier);
}

tree::TerminalNode* qasm3Parser::EqualsAssignmentListContext::Identifier(size_t i) {
  return getToken(qasm3Parser::Identifier, i);
}

std::vector<qasm3Parser::EqualsExpressionContext *> qasm3Parser::EqualsAssignmentListContext::equalsExpression() {
  return getRuleContexts<qasm3Parser::EqualsExpressionContext>();
}

qasm3Parser::EqualsExpressionContext* qasm3Parser::EqualsAssignmentListContext::equalsExpression(size_t i) {
  return getRuleContext<qasm3Parser::EqualsExpressionContext>(i);
}

std::vector<tree::TerminalNode *> qasm3Parser::EqualsAssignmentListContext::COMMA() {
  return getTokens(qasm3Parser::COMMA);
}

tree::TerminalNode* qasm3Parser::EqualsAssignmentListContext::COMMA(size_t i) {
  return getToken(qasm3Parser::COMMA, i);
}


size_t qasm3Parser::EqualsAssignmentListContext::getRuleIndex() const {
  return qasm3Parser::RuleEqualsAssignmentList;
}

void qasm3Parser::EqualsAssignmentListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEqualsAssignmentList(this);
}

void qasm3Parser::EqualsAssignmentListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEqualsAssignmentList(this);
}


antlrcpp::Any qasm3Parser::EqualsAssignmentListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitEqualsAssignmentList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::EqualsAssignmentListContext* qasm3Parser::equalsAssignmentList() {
  EqualsAssignmentListContext *_localctx = _tracker.createInstance<EqualsAssignmentListContext>(_ctx, getState());
  enterRule(_localctx, 150, qasm3Parser::RuleEqualsAssignmentList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(759);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 64, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(753);
        match(qasm3Parser::Identifier);
        setState(754);
        equalsExpression();
        setState(755);
        match(qasm3Parser::COMMA); 
      }
      setState(761);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 64, _ctx);
    }
    setState(762);
    match(qasm3Parser::Identifier);
    setState(763);
    equalsExpression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MembershipTestContext ------------------------------------------------------------------

qasm3Parser::MembershipTestContext::MembershipTestContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::MembershipTestContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::SetDeclarationContext* qasm3Parser::MembershipTestContext::setDeclaration() {
  return getRuleContext<qasm3Parser::SetDeclarationContext>(0);
}


size_t qasm3Parser::MembershipTestContext::getRuleIndex() const {
  return qasm3Parser::RuleMembershipTest;
}

void qasm3Parser::MembershipTestContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMembershipTest(this);
}

void qasm3Parser::MembershipTestContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMembershipTest(this);
}


antlrcpp::Any qasm3Parser::MembershipTestContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitMembershipTest(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::MembershipTestContext* qasm3Parser::membershipTest() {
  MembershipTestContext *_localctx = _tracker.createInstance<MembershipTestContext>(_ctx, getState());
  enterRule(_localctx, 152, qasm3Parser::RuleMembershipTest);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(765);
    match(qasm3Parser::Identifier);
    setState(766);
    match(qasm3Parser::T__74);
    setState(767);
    setDeclaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SetDeclarationContext ------------------------------------------------------------------

qasm3Parser::SetDeclarationContext::SetDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::SetDeclarationContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

qasm3Parser::ExpressionListContext* qasm3Parser::SetDeclarationContext::expressionList() {
  return getRuleContext<qasm3Parser::ExpressionListContext>(0);
}

tree::TerminalNode* qasm3Parser::SetDeclarationContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}

qasm3Parser::RangeDefinitionContext* qasm3Parser::SetDeclarationContext::rangeDefinition() {
  return getRuleContext<qasm3Parser::RangeDefinitionContext>(0);
}

tree::TerminalNode* qasm3Parser::SetDeclarationContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}


size_t qasm3Parser::SetDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleSetDeclaration;
}

void qasm3Parser::SetDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSetDeclaration(this);
}

void qasm3Parser::SetDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSetDeclaration(this);
}


antlrcpp::Any qasm3Parser::SetDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitSetDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::SetDeclarationContext* qasm3Parser::setDeclaration() {
  SetDeclarationContext *_localctx = _tracker.createInstance<SetDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 154, qasm3Parser::RuleSetDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(775);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::LBRACE: {
        enterOuterAlt(_localctx, 1);
        setState(769);
        match(qasm3Parser::LBRACE);
        setState(770);
        expressionList();
        setState(771);
        match(qasm3Parser::RBRACE);
        break;
      }

      case qasm3Parser::LBRACKET: {
        enterOuterAlt(_localctx, 2);
        setState(773);
        rangeDefinition();
        break;
      }

      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 3);
        setState(774);
        match(qasm3Parser::Identifier);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ProgramBlockContext ------------------------------------------------------------------

qasm3Parser::ProgramBlockContext::ProgramBlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasm3Parser::StatementContext *> qasm3Parser::ProgramBlockContext::statement() {
  return getRuleContexts<qasm3Parser::StatementContext>();
}

qasm3Parser::StatementContext* qasm3Parser::ProgramBlockContext::statement(size_t i) {
  return getRuleContext<qasm3Parser::StatementContext>(i);
}

tree::TerminalNode* qasm3Parser::ProgramBlockContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

tree::TerminalNode* qasm3Parser::ProgramBlockContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}


size_t qasm3Parser::ProgramBlockContext::getRuleIndex() const {
  return qasm3Parser::RuleProgramBlock;
}

void qasm3Parser::ProgramBlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgramBlock(this);
}

void qasm3Parser::ProgramBlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgramBlock(this);
}


antlrcpp::Any qasm3Parser::ProgramBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitProgramBlock(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ProgramBlockContext* qasm3Parser::programBlock() {
  ProgramBlockContext *_localctx = _tracker.createInstance<ProgramBlockContext>(_ctx, getState());
  enterRule(_localctx, 156, qasm3Parser::RuleProgramBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(786);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__2:
      case qasm3Parser::T__4:
      case qasm3Parser::T__7:
      case qasm3Parser::T__8:
      case qasm3Parser::T__9:
      case qasm3Parser::T__10:
      case qasm3Parser::T__11:
      case qasm3Parser::T__12:
      case qasm3Parser::T__13:
      case qasm3Parser::T__14:
      case qasm3Parser::T__15:
      case qasm3Parser::T__16:
      case qasm3Parser::T__17:
      case qasm3Parser::T__18:
      case qasm3Parser::T__21:
      case qasm3Parser::T__22:
      case qasm3Parser::T__23:
      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26:
      case qasm3Parser::T__27:
      case qasm3Parser::T__28:
      case qasm3Parser::T__30:
      case qasm3Parser::T__31:
      case qasm3Parser::T__32:
      case qasm3Parser::T__51:
      case qasm3Parser::T__52:
      case qasm3Parser::T__53:
      case qasm3Parser::T__54:
      case qasm3Parser::T__55:
      case qasm3Parser::T__56:
      case qasm3Parser::T__57:
      case qasm3Parser::T__58:
      case qasm3Parser::T__59:
      case qasm3Parser::T__60:
      case qasm3Parser::T__61:
      case qasm3Parser::T__62:
      case qasm3Parser::T__63:
      case qasm3Parser::T__75:
      case qasm3Parser::T__77:
      case qasm3Parser::T__78:
      case qasm3Parser::T__80:
      case qasm3Parser::T__81:
      case qasm3Parser::T__82:
      case qasm3Parser::T__85:
      case qasm3Parser::T__87:
      case qasm3Parser::T__88:
      case qasm3Parser::T__89:
      case qasm3Parser::T__90:
      case qasm3Parser::T__91:
      case qasm3Parser::T__92:
      case qasm3Parser::T__93:
      case qasm3Parser::LPAREN:
      case qasm3Parser::MINUS:
      case qasm3Parser::Constant:
      case qasm3Parser::Integer:
      case qasm3Parser::Identifier:
      case qasm3Parser::RealNumber:
      case qasm3Parser::TimingLiteral:
      case qasm3Parser::StringLiteral: {
        enterOuterAlt(_localctx, 1);
        setState(777);
        statement();
        break;
      }

      case qasm3Parser::LBRACE: {
        enterOuterAlt(_localctx, 2);
        setState(778);
        match(qasm3Parser::LBRACE);
        setState(782);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << qasm3Parser::T__2)
          | (1ULL << qasm3Parser::T__4)
          | (1ULL << qasm3Parser::T__7)
          | (1ULL << qasm3Parser::T__8)
          | (1ULL << qasm3Parser::T__9)
          | (1ULL << qasm3Parser::T__10)
          | (1ULL << qasm3Parser::T__11)
          | (1ULL << qasm3Parser::T__12)
          | (1ULL << qasm3Parser::T__13)
          | (1ULL << qasm3Parser::T__14)
          | (1ULL << qasm3Parser::T__15)
          | (1ULL << qasm3Parser::T__16)
          | (1ULL << qasm3Parser::T__17)
          | (1ULL << qasm3Parser::T__18)
          | (1ULL << qasm3Parser::T__21)
          | (1ULL << qasm3Parser::T__22)
          | (1ULL << qasm3Parser::T__23)
          | (1ULL << qasm3Parser::T__24)
          | (1ULL << qasm3Parser::T__25)
          | (1ULL << qasm3Parser::T__26)
          | (1ULL << qasm3Parser::T__27)
          | (1ULL << qasm3Parser::T__28)
          | (1ULL << qasm3Parser::T__30)
          | (1ULL << qasm3Parser::T__31)
          | (1ULL << qasm3Parser::T__32)
          | (1ULL << qasm3Parser::T__51)
          | (1ULL << qasm3Parser::T__52)
          | (1ULL << qasm3Parser::T__53)
          | (1ULL << qasm3Parser::T__54)
          | (1ULL << qasm3Parser::T__55)
          | (1ULL << qasm3Parser::T__56)
          | (1ULL << qasm3Parser::T__57)
          | (1ULL << qasm3Parser::T__58)
          | (1ULL << qasm3Parser::T__59)
          | (1ULL << qasm3Parser::T__60)
          | (1ULL << qasm3Parser::T__61)
          | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
          | (1ULL << (qasm3Parser::T__75 - 64))
          | (1ULL << (qasm3Parser::T__77 - 64))
          | (1ULL << (qasm3Parser::T__78 - 64))
          | (1ULL << (qasm3Parser::T__80 - 64))
          | (1ULL << (qasm3Parser::T__81 - 64))
          | (1ULL << (qasm3Parser::T__82 - 64))
          | (1ULL << (qasm3Parser::T__85 - 64))
          | (1ULL << (qasm3Parser::T__87 - 64))
          | (1ULL << (qasm3Parser::T__88 - 64))
          | (1ULL << (qasm3Parser::T__89 - 64))
          | (1ULL << (qasm3Parser::T__90 - 64))
          | (1ULL << (qasm3Parser::T__91 - 64))
          | (1ULL << (qasm3Parser::T__92 - 64))
          | (1ULL << (qasm3Parser::T__93 - 64))
          | (1ULL << (qasm3Parser::LPAREN - 64))
          | (1ULL << (qasm3Parser::MINUS - 64))
          | (1ULL << (qasm3Parser::Constant - 64))
          | (1ULL << (qasm3Parser::Integer - 64))
          | (1ULL << (qasm3Parser::Identifier - 64))
          | (1ULL << (qasm3Parser::RealNumber - 64))
          | (1ULL << (qasm3Parser::TimingLiteral - 64))
          | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
          setState(779);
          statement();
          setState(784);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(785);
        match(qasm3Parser::RBRACE);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BranchingStatementContext ------------------------------------------------------------------

qasm3Parser::BranchingStatementContext::BranchingStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::BranchingStatementContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

qasm3Parser::BooleanExpressionContext* qasm3Parser::BranchingStatementContext::booleanExpression() {
  return getRuleContext<qasm3Parser::BooleanExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::BranchingStatementContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

std::vector<qasm3Parser::ProgramBlockContext *> qasm3Parser::BranchingStatementContext::programBlock() {
  return getRuleContexts<qasm3Parser::ProgramBlockContext>();
}

qasm3Parser::ProgramBlockContext* qasm3Parser::BranchingStatementContext::programBlock(size_t i) {
  return getRuleContext<qasm3Parser::ProgramBlockContext>(i);
}


size_t qasm3Parser::BranchingStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleBranchingStatement;
}

void qasm3Parser::BranchingStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBranchingStatement(this);
}

void qasm3Parser::BranchingStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBranchingStatement(this);
}


antlrcpp::Any qasm3Parser::BranchingStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitBranchingStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::BranchingStatementContext* qasm3Parser::branchingStatement() {
  BranchingStatementContext *_localctx = _tracker.createInstance<BranchingStatementContext>(_ctx, getState());
  enterRule(_localctx, 158, qasm3Parser::RuleBranchingStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(788);
    match(qasm3Parser::T__75);
    setState(789);
    match(qasm3Parser::LPAREN);
    setState(790);
    booleanExpression(0);
    setState(791);
    match(qasm3Parser::RPAREN);
    setState(792);
    programBlock();
    setState(795);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 68, _ctx)) {
    case 1: {
      setState(793);
      match(qasm3Parser::T__76);
      setState(794);
      programBlock();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LoopSignatureContext ------------------------------------------------------------------

qasm3Parser::LoopSignatureContext::LoopSignatureContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::MembershipTestContext* qasm3Parser::LoopSignatureContext::membershipTest() {
  return getRuleContext<qasm3Parser::MembershipTestContext>(0);
}

tree::TerminalNode* qasm3Parser::LoopSignatureContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

qasm3Parser::BooleanExpressionContext* qasm3Parser::LoopSignatureContext::booleanExpression() {
  return getRuleContext<qasm3Parser::BooleanExpressionContext>(0);
}

tree::TerminalNode* qasm3Parser::LoopSignatureContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}


size_t qasm3Parser::LoopSignatureContext::getRuleIndex() const {
  return qasm3Parser::RuleLoopSignature;
}

void qasm3Parser::LoopSignatureContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoopSignature(this);
}

void qasm3Parser::LoopSignatureContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoopSignature(this);
}


antlrcpp::Any qasm3Parser::LoopSignatureContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitLoopSignature(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::LoopSignatureContext* qasm3Parser::loopSignature() {
  LoopSignatureContext *_localctx = _tracker.createInstance<LoopSignatureContext>(_ctx, getState());
  enterRule(_localctx, 160, qasm3Parser::RuleLoopSignature);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(804);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__77: {
        enterOuterAlt(_localctx, 1);
        setState(797);
        match(qasm3Parser::T__77);
        setState(798);
        membershipTest();
        break;
      }

      case qasm3Parser::T__78: {
        enterOuterAlt(_localctx, 2);
        setState(799);
        match(qasm3Parser::T__78);
        setState(800);
        match(qasm3Parser::LPAREN);
        setState(801);
        booleanExpression(0);
        setState(802);
        match(qasm3Parser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LoopStatementContext ------------------------------------------------------------------

qasm3Parser::LoopStatementContext::LoopStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::LoopSignatureContext* qasm3Parser::LoopStatementContext::loopSignature() {
  return getRuleContext<qasm3Parser::LoopSignatureContext>(0);
}

qasm3Parser::ProgramBlockContext* qasm3Parser::LoopStatementContext::programBlock() {
  return getRuleContext<qasm3Parser::ProgramBlockContext>(0);
}


size_t qasm3Parser::LoopStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleLoopStatement;
}

void qasm3Parser::LoopStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoopStatement(this);
}

void qasm3Parser::LoopStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoopStatement(this);
}


antlrcpp::Any qasm3Parser::LoopStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitLoopStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::LoopStatementContext* qasm3Parser::loopStatement() {
  LoopStatementContext *_localctx = _tracker.createInstance<LoopStatementContext>(_ctx, getState());
  enterRule(_localctx, 162, qasm3Parser::RuleLoopStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(806);
    loopSignature();
    setState(807);
    programBlock();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CLikeLoopStatementContext ------------------------------------------------------------------

qasm3Parser::CLikeLoopStatementContext::CLikeLoopStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> qasm3Parser::CLikeLoopStatementContext::LPAREN() {
  return getTokens(qasm3Parser::LPAREN);
}

tree::TerminalNode* qasm3Parser::CLikeLoopStatementContext::LPAREN(size_t i) {
  return getToken(qasm3Parser::LPAREN, i);
}

qasm3Parser::ClassicalTypeContext* qasm3Parser::CLikeLoopStatementContext::classicalType() {
  return getRuleContext<qasm3Parser::ClassicalTypeContext>(0);
}

tree::TerminalNode* qasm3Parser::CLikeLoopStatementContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::EqualsExpressionContext* qasm3Parser::CLikeLoopStatementContext::equalsExpression() {
  return getRuleContext<qasm3Parser::EqualsExpressionContext>(0);
}

std::vector<tree::TerminalNode *> qasm3Parser::CLikeLoopStatementContext::SEMICOLON() {
  return getTokens(qasm3Parser::SEMICOLON);
}

tree::TerminalNode* qasm3Parser::CLikeLoopStatementContext::SEMICOLON(size_t i) {
  return getToken(qasm3Parser::SEMICOLON, i);
}

qasm3Parser::BooleanExpressionContext* qasm3Parser::CLikeLoopStatementContext::booleanExpression() {
  return getRuleContext<qasm3Parser::BooleanExpressionContext>(0);
}

qasm3Parser::ExpressionContext* qasm3Parser::CLikeLoopStatementContext::expression() {
  return getRuleContext<qasm3Parser::ExpressionContext>(0);
}

std::vector<tree::TerminalNode *> qasm3Parser::CLikeLoopStatementContext::RPAREN() {
  return getTokens(qasm3Parser::RPAREN);
}

tree::TerminalNode* qasm3Parser::CLikeLoopStatementContext::RPAREN(size_t i) {
  return getToken(qasm3Parser::RPAREN, i);
}

tree::TerminalNode* qasm3Parser::CLikeLoopStatementContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

qasm3Parser::ProgramBlockContext* qasm3Parser::CLikeLoopStatementContext::programBlock() {
  return getRuleContext<qasm3Parser::ProgramBlockContext>(0);
}

tree::TerminalNode* qasm3Parser::CLikeLoopStatementContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}

tree::TerminalNode* qasm3Parser::CLikeLoopStatementContext::COLON() {
  return getToken(qasm3Parser::COLON, 0);
}


size_t qasm3Parser::CLikeLoopStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleCLikeLoopStatement;
}

void qasm3Parser::CLikeLoopStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCLikeLoopStatement(this);
}

void qasm3Parser::CLikeLoopStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCLikeLoopStatement(this);
}


antlrcpp::Any qasm3Parser::CLikeLoopStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCLikeLoopStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::CLikeLoopStatementContext* qasm3Parser::cLikeLoopStatement() {
  CLikeLoopStatementContext *_localctx = _tracker.createInstance<CLikeLoopStatementContext>(_ctx, getState());
  enterRule(_localctx, 164, qasm3Parser::RuleCLikeLoopStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(837);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 70, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(809);
      match(qasm3Parser::T__77);
      setState(810);
      match(qasm3Parser::LPAREN);
      setState(811);
      classicalType();
      setState(812);
      match(qasm3Parser::Identifier);
      setState(813);
      equalsExpression();
      setState(814);
      match(qasm3Parser::SEMICOLON);
      setState(815);
      booleanExpression(0);
      setState(816);
      match(qasm3Parser::SEMICOLON);
      setState(817);
      expression(0);
      setState(818);
      match(qasm3Parser::RPAREN);
      setState(819);
      match(qasm3Parser::LBRACE);
      setState(820);
      programBlock();
      setState(821);
      match(qasm3Parser::RBRACE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(823);
      match(qasm3Parser::T__77);
      setState(824);
      match(qasm3Parser::LPAREN);
      setState(825);
      classicalType();
      setState(826);
      match(qasm3Parser::Identifier);
      setState(827);
      match(qasm3Parser::COLON);
      setState(828);
      match(qasm3Parser::T__79);
      setState(829);
      match(qasm3Parser::LPAREN);
      setState(830);
      expression(0);
      setState(831);
      match(qasm3Parser::RPAREN);
      setState(832);
      match(qasm3Parser::RPAREN);
      setState(833);
      match(qasm3Parser::LBRACE);
      setState(834);
      programBlock();
      setState(835);
      match(qasm3Parser::RBRACE);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ControlDirectiveStatementContext ------------------------------------------------------------------

qasm3Parser::ControlDirectiveStatementContext::ControlDirectiveStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ControlDirectiveContext* qasm3Parser::ControlDirectiveStatementContext::controlDirective() {
  return getRuleContext<qasm3Parser::ControlDirectiveContext>(0);
}

tree::TerminalNode* qasm3Parser::ControlDirectiveStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::ControlDirectiveStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleControlDirectiveStatement;
}

void qasm3Parser::ControlDirectiveStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterControlDirectiveStatement(this);
}

void qasm3Parser::ControlDirectiveStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitControlDirectiveStatement(this);
}


antlrcpp::Any qasm3Parser::ControlDirectiveStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitControlDirectiveStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ControlDirectiveStatementContext* qasm3Parser::controlDirectiveStatement() {
  ControlDirectiveStatementContext *_localctx = _tracker.createInstance<ControlDirectiveStatementContext>(_ctx, getState());
  enterRule(_localctx, 166, qasm3Parser::RuleControlDirectiveStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(839);
    controlDirective();
    setState(840);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ControlDirectiveContext ------------------------------------------------------------------

qasm3Parser::ControlDirectiveContext::ControlDirectiveContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::ControlDirectiveContext::getRuleIndex() const {
  return qasm3Parser::RuleControlDirective;
}

void qasm3Parser::ControlDirectiveContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterControlDirective(this);
}

void qasm3Parser::ControlDirectiveContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitControlDirective(this);
}


antlrcpp::Any qasm3Parser::ControlDirectiveContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitControlDirective(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ControlDirectiveContext* qasm3Parser::controlDirective() {
  ControlDirectiveContext *_localctx = _tracker.createInstance<ControlDirectiveContext>(_ctx, getState());
  enterRule(_localctx, 168, qasm3Parser::RuleControlDirective);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(842);
    _la = _input->LA(1);
    if (!(((((_la - 81) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 81)) & ((1ULL << (qasm3Parser::T__80 - 81))
      | (1ULL << (qasm3Parser::T__81 - 81))
      | (1ULL << (qasm3Parser::T__82 - 81)))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- KernelDeclarationContext ------------------------------------------------------------------

qasm3Parser::KernelDeclarationContext::KernelDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::KernelDeclarationContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

tree::TerminalNode* qasm3Parser::KernelDeclarationContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}

tree::TerminalNode* qasm3Parser::KernelDeclarationContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::KernelDeclarationContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

qasm3Parser::ReturnSignatureContext* qasm3Parser::KernelDeclarationContext::returnSignature() {
  return getRuleContext<qasm3Parser::ReturnSignatureContext>(0);
}

qasm3Parser::ClassicalTypeContext* qasm3Parser::KernelDeclarationContext::classicalType() {
  return getRuleContext<qasm3Parser::ClassicalTypeContext>(0);
}

qasm3Parser::ClassicalTypeListContext* qasm3Parser::KernelDeclarationContext::classicalTypeList() {
  return getRuleContext<qasm3Parser::ClassicalTypeListContext>(0);
}


size_t qasm3Parser::KernelDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleKernelDeclaration;
}

void qasm3Parser::KernelDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterKernelDeclaration(this);
}

void qasm3Parser::KernelDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitKernelDeclaration(this);
}


antlrcpp::Any qasm3Parser::KernelDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitKernelDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::KernelDeclarationContext* qasm3Parser::kernelDeclaration() {
  KernelDeclarationContext *_localctx = _tracker.createInstance<KernelDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 170, qasm3Parser::RuleKernelDeclaration);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(844);
    match(qasm3Parser::T__83);
    setState(845);
    match(qasm3Parser::Identifier);
    setState(851);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(846);
      match(qasm3Parser::LPAREN);
      setState(848);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__15)
        | (1ULL << qasm3Parser::T__16))) != 0) || _la == qasm3Parser::T__87

      || _la == qasm3Parser::T__88) {
        setState(847);
        classicalTypeList();
      }
      setState(850);
      match(qasm3Parser::RPAREN);
    }
    setState(854);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::ARROW) {
      setState(853);
      returnSignature();
    }
    setState(857);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__15)
      | (1ULL << qasm3Parser::T__16))) != 0) || _la == qasm3Parser::T__87

    || _la == qasm3Parser::T__88) {
      setState(856);
      classicalType();
    }
    setState(859);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- KernelCallContext ------------------------------------------------------------------

qasm3Parser::KernelCallContext::KernelCallContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::KernelCallContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

tree::TerminalNode* qasm3Parser::KernelCallContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::KernelCallContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

qasm3Parser::ExpressionListContext* qasm3Parser::KernelCallContext::expressionList() {
  return getRuleContext<qasm3Parser::ExpressionListContext>(0);
}


size_t qasm3Parser::KernelCallContext::getRuleIndex() const {
  return qasm3Parser::RuleKernelCall;
}

void qasm3Parser::KernelCallContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterKernelCall(this);
}

void qasm3Parser::KernelCallContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitKernelCall(this);
}


antlrcpp::Any qasm3Parser::KernelCallContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitKernelCall(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::KernelCallContext* qasm3Parser::kernelCall() {
  KernelCallContext *_localctx = _tracker.createInstance<KernelCallContext>(_ctx, getState());
  enterRule(_localctx, 172, qasm3Parser::RuleKernelCall);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(861);
    match(qasm3Parser::Identifier);
    setState(862);
    match(qasm3Parser::LPAREN);
    setState(864);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__15)
      | (1ULL << qasm3Parser::T__16)
      | (1ULL << qasm3Parser::T__31)
      | (1ULL << qasm3Parser::T__32)
      | (1ULL << qasm3Parser::T__51)
      | (1ULL << qasm3Parser::T__52)
      | (1ULL << qasm3Parser::T__53)
      | (1ULL << qasm3Parser::T__54)
      | (1ULL << qasm3Parser::T__55)
      | (1ULL << qasm3Parser::T__56)
      | (1ULL << qasm3Parser::T__57)
      | (1ULL << qasm3Parser::T__58)
      | (1ULL << qasm3Parser::T__59)
      | (1ULL << qasm3Parser::T__60)
      | (1ULL << qasm3Parser::T__61)
      | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
      | (1ULL << (qasm3Parser::T__87 - 64))
      | (1ULL << (qasm3Parser::T__88 - 64))
      | (1ULL << (qasm3Parser::T__91 - 64))
      | (1ULL << (qasm3Parser::LPAREN - 64))
      | (1ULL << (qasm3Parser::MINUS - 64))
      | (1ULL << (qasm3Parser::Constant - 64))
      | (1ULL << (qasm3Parser::Integer - 64))
      | (1ULL << (qasm3Parser::Identifier - 64))
      | (1ULL << (qasm3Parser::RealNumber - 64))
      | (1ULL << (qasm3Parser::TimingLiteral - 64))
      | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
      setState(863);
      expressionList();
    }
    setState(866);
    match(qasm3Parser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SubroutineDefinitionContext ------------------------------------------------------------------

qasm3Parser::SubroutineDefinitionContext::SubroutineDefinitionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::SubroutineDefinitionContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::SubroutineBlockContext* qasm3Parser::SubroutineDefinitionContext::subroutineBlock() {
  return getRuleContext<qasm3Parser::SubroutineBlockContext>(0);
}

tree::TerminalNode* qasm3Parser::SubroutineDefinitionContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::SubroutineDefinitionContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

qasm3Parser::QuantumArgumentListContext* qasm3Parser::SubroutineDefinitionContext::quantumArgumentList() {
  return getRuleContext<qasm3Parser::QuantumArgumentListContext>(0);
}

qasm3Parser::ReturnSignatureContext* qasm3Parser::SubroutineDefinitionContext::returnSignature() {
  return getRuleContext<qasm3Parser::ReturnSignatureContext>(0);
}

qasm3Parser::ClassicalArgumentListContext* qasm3Parser::SubroutineDefinitionContext::classicalArgumentList() {
  return getRuleContext<qasm3Parser::ClassicalArgumentListContext>(0);
}


size_t qasm3Parser::SubroutineDefinitionContext::getRuleIndex() const {
  return qasm3Parser::RuleSubroutineDefinition;
}

void qasm3Parser::SubroutineDefinitionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubroutineDefinition(this);
}

void qasm3Parser::SubroutineDefinitionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubroutineDefinition(this);
}


antlrcpp::Any qasm3Parser::SubroutineDefinitionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitSubroutineDefinition(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::SubroutineDefinitionContext* qasm3Parser::subroutineDefinition() {
  SubroutineDefinitionContext *_localctx = _tracker.createInstance<SubroutineDefinitionContext>(_ctx, getState());
  enterRule(_localctx, 174, qasm3Parser::RuleSubroutineDefinition);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(868);
    match(qasm3Parser::T__84);
    setState(869);
    match(qasm3Parser::Identifier);
    setState(875);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(870);
      match(qasm3Parser::LPAREN);
      setState(872);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__15)
        | (1ULL << qasm3Parser::T__16))) != 0) || _la == qasm3Parser::T__87

      || _la == qasm3Parser::T__88) {
        setState(871);
        classicalArgumentList();
      }
      setState(874);
      match(qasm3Parser::RPAREN);
    }
    setState(878);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::T__5

    || _la == qasm3Parser::T__6) {
      setState(877);
      quantumArgumentList();
    }
    setState(881);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::ARROW) {
      setState(880);
      returnSignature();
    }
    setState(883);
    subroutineBlock();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReturnStatementContext ------------------------------------------------------------------

qasm3Parser::ReturnStatementContext::ReturnStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::StatementContext* qasm3Parser::ReturnStatementContext::statement() {
  return getRuleContext<qasm3Parser::StatementContext>(0);
}


size_t qasm3Parser::ReturnStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleReturnStatement;
}

void qasm3Parser::ReturnStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReturnStatement(this);
}

void qasm3Parser::ReturnStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReturnStatement(this);
}


antlrcpp::Any qasm3Parser::ReturnStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitReturnStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::ReturnStatementContext* qasm3Parser::returnStatement() {
  ReturnStatementContext *_localctx = _tracker.createInstance<ReturnStatementContext>(_ctx, getState());
  enterRule(_localctx, 176, qasm3Parser::RuleReturnStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(885);
    match(qasm3Parser::T__85);
    setState(886);
    statement();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SubroutineBlockContext ------------------------------------------------------------------

qasm3Parser::SubroutineBlockContext::SubroutineBlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::SubroutineBlockContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

tree::TerminalNode* qasm3Parser::SubroutineBlockContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}

std::vector<qasm3Parser::StatementContext *> qasm3Parser::SubroutineBlockContext::statement() {
  return getRuleContexts<qasm3Parser::StatementContext>();
}

qasm3Parser::StatementContext* qasm3Parser::SubroutineBlockContext::statement(size_t i) {
  return getRuleContext<qasm3Parser::StatementContext>(i);
}

qasm3Parser::ReturnStatementContext* qasm3Parser::SubroutineBlockContext::returnStatement() {
  return getRuleContext<qasm3Parser::ReturnStatementContext>(0);
}

tree::TerminalNode* qasm3Parser::SubroutineBlockContext::EXTERN() {
  return getToken(qasm3Parser::EXTERN, 0);
}

tree::TerminalNode* qasm3Parser::SubroutineBlockContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::SubroutineBlockContext::getRuleIndex() const {
  return qasm3Parser::RuleSubroutineBlock;
}

void qasm3Parser::SubroutineBlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubroutineBlock(this);
}

void qasm3Parser::SubroutineBlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubroutineBlock(this);
}


antlrcpp::Any qasm3Parser::SubroutineBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitSubroutineBlock(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::SubroutineBlockContext* qasm3Parser::subroutineBlock() {
  SubroutineBlockContext *_localctx = _tracker.createInstance<SubroutineBlockContext>(_ctx, getState());
  enterRule(_localctx, 178, qasm3Parser::RuleSubroutineBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(901);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::LBRACE: {
        enterOuterAlt(_localctx, 1);
        setState(888);
        match(qasm3Parser::LBRACE);
        setState(892);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 80, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(889);
            statement(); 
          }
          setState(894);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 80, _ctx);
        }
        setState(896);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == qasm3Parser::T__85) {
          setState(895);
          returnStatement();
        }
        setState(898);
        match(qasm3Parser::RBRACE);
        break;
      }

      case qasm3Parser::EXTERN: {
        enterOuterAlt(_localctx, 2);
        setState(899);
        match(qasm3Parser::EXTERN);
        setState(900);
        match(qasm3Parser::SEMICOLON);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SubroutineCallContext ------------------------------------------------------------------

qasm3Parser::SubroutineCallContext::SubroutineCallContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::SubroutineCallContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

std::vector<qasm3Parser::ExpressionListContext *> qasm3Parser::SubroutineCallContext::expressionList() {
  return getRuleContexts<qasm3Parser::ExpressionListContext>();
}

qasm3Parser::ExpressionListContext* qasm3Parser::SubroutineCallContext::expressionList(size_t i) {
  return getRuleContext<qasm3Parser::ExpressionListContext>(i);
}

tree::TerminalNode* qasm3Parser::SubroutineCallContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::SubroutineCallContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}


size_t qasm3Parser::SubroutineCallContext::getRuleIndex() const {
  return qasm3Parser::RuleSubroutineCall;
}

void qasm3Parser::SubroutineCallContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubroutineCall(this);
}

void qasm3Parser::SubroutineCallContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubroutineCall(this);
}


antlrcpp::Any qasm3Parser::SubroutineCallContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitSubroutineCall(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::SubroutineCallContext* qasm3Parser::subroutineCall() {
  SubroutineCallContext *_localctx = _tracker.createInstance<SubroutineCallContext>(_ctx, getState());
  enterRule(_localctx, 180, qasm3Parser::RuleSubroutineCall);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(903);
    match(qasm3Parser::Identifier);

    setState(904);
    match(qasm3Parser::LPAREN);
    setState(906);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__15)
      | (1ULL << qasm3Parser::T__16)
      | (1ULL << qasm3Parser::T__31)
      | (1ULL << qasm3Parser::T__32)
      | (1ULL << qasm3Parser::T__51)
      | (1ULL << qasm3Parser::T__52)
      | (1ULL << qasm3Parser::T__53)
      | (1ULL << qasm3Parser::T__54)
      | (1ULL << qasm3Parser::T__55)
      | (1ULL << qasm3Parser::T__56)
      | (1ULL << qasm3Parser::T__57)
      | (1ULL << qasm3Parser::T__58)
      | (1ULL << qasm3Parser::T__59)
      | (1ULL << qasm3Parser::T__60)
      | (1ULL << qasm3Parser::T__61)
      | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
      | (1ULL << (qasm3Parser::T__87 - 64))
      | (1ULL << (qasm3Parser::T__88 - 64))
      | (1ULL << (qasm3Parser::T__91 - 64))
      | (1ULL << (qasm3Parser::LPAREN - 64))
      | (1ULL << (qasm3Parser::MINUS - 64))
      | (1ULL << (qasm3Parser::Constant - 64))
      | (1ULL << (qasm3Parser::Integer - 64))
      | (1ULL << (qasm3Parser::Identifier - 64))
      | (1ULL << (qasm3Parser::RealNumber - 64))
      | (1ULL << (qasm3Parser::TimingLiteral - 64))
      | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
      setState(905);
      expressionList();
    }
    setState(908);
    match(qasm3Parser::RPAREN);
    setState(910);
    expressionList();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PragmaContext ------------------------------------------------------------------

qasm3Parser::PragmaContext::PragmaContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::PragmaContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

tree::TerminalNode* qasm3Parser::PragmaContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}

std::vector<qasm3Parser::StatementContext *> qasm3Parser::PragmaContext::statement() {
  return getRuleContexts<qasm3Parser::StatementContext>();
}

qasm3Parser::StatementContext* qasm3Parser::PragmaContext::statement(size_t i) {
  return getRuleContext<qasm3Parser::StatementContext>(i);
}


size_t qasm3Parser::PragmaContext::getRuleIndex() const {
  return qasm3Parser::RulePragma;
}

void qasm3Parser::PragmaContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPragma(this);
}

void qasm3Parser::PragmaContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPragma(this);
}


antlrcpp::Any qasm3Parser::PragmaContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitPragma(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::PragmaContext* qasm3Parser::pragma() {
  PragmaContext *_localctx = _tracker.createInstance<PragmaContext>(_ctx, getState());
  enterRule(_localctx, 182, qasm3Parser::RulePragma);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(912);
    match(qasm3Parser::T__86);
    setState(913);
    match(qasm3Parser::LBRACE);
    setState(917);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__2)
      | (1ULL << qasm3Parser::T__4)
      | (1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__15)
      | (1ULL << qasm3Parser::T__16)
      | (1ULL << qasm3Parser::T__17)
      | (1ULL << qasm3Parser::T__18)
      | (1ULL << qasm3Parser::T__21)
      | (1ULL << qasm3Parser::T__22)
      | (1ULL << qasm3Parser::T__23)
      | (1ULL << qasm3Parser::T__24)
      | (1ULL << qasm3Parser::T__25)
      | (1ULL << qasm3Parser::T__26)
      | (1ULL << qasm3Parser::T__27)
      | (1ULL << qasm3Parser::T__28)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__31)
      | (1ULL << qasm3Parser::T__32)
      | (1ULL << qasm3Parser::T__51)
      | (1ULL << qasm3Parser::T__52)
      | (1ULL << qasm3Parser::T__53)
      | (1ULL << qasm3Parser::T__54)
      | (1ULL << qasm3Parser::T__55)
      | (1ULL << qasm3Parser::T__56)
      | (1ULL << qasm3Parser::T__57)
      | (1ULL << qasm3Parser::T__58)
      | (1ULL << qasm3Parser::T__59)
      | (1ULL << qasm3Parser::T__60)
      | (1ULL << qasm3Parser::T__61)
      | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
      | (1ULL << (qasm3Parser::T__75 - 64))
      | (1ULL << (qasm3Parser::T__77 - 64))
      | (1ULL << (qasm3Parser::T__78 - 64))
      | (1ULL << (qasm3Parser::T__80 - 64))
      | (1ULL << (qasm3Parser::T__81 - 64))
      | (1ULL << (qasm3Parser::T__82 - 64))
      | (1ULL << (qasm3Parser::T__85 - 64))
      | (1ULL << (qasm3Parser::T__87 - 64))
      | (1ULL << (qasm3Parser::T__88 - 64))
      | (1ULL << (qasm3Parser::T__89 - 64))
      | (1ULL << (qasm3Parser::T__90 - 64))
      | (1ULL << (qasm3Parser::T__91 - 64))
      | (1ULL << (qasm3Parser::T__92 - 64))
      | (1ULL << (qasm3Parser::T__93 - 64))
      | (1ULL << (qasm3Parser::LPAREN - 64))
      | (1ULL << (qasm3Parser::MINUS - 64))
      | (1ULL << (qasm3Parser::Constant - 64))
      | (1ULL << (qasm3Parser::Integer - 64))
      | (1ULL << (qasm3Parser::Identifier - 64))
      | (1ULL << (qasm3Parser::RealNumber - 64))
      | (1ULL << (qasm3Parser::TimingLiteral - 64))
      | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
      setState(914);
      statement();
      setState(919);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(920);
    match(qasm3Parser::RBRACE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TimingTypeContext ------------------------------------------------------------------

qasm3Parser::TimingTypeContext::TimingTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::TimingTypeContext::Integer() {
  return getToken(qasm3Parser::Integer, 0);
}


size_t qasm3Parser::TimingTypeContext::getRuleIndex() const {
  return qasm3Parser::RuleTimingType;
}

void qasm3Parser::TimingTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTimingType(this);
}

void qasm3Parser::TimingTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTimingType(this);
}


antlrcpp::Any qasm3Parser::TimingTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitTimingType(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::TimingTypeContext* qasm3Parser::timingType() {
  TimingTypeContext *_localctx = _tracker.createInstance<TimingTypeContext>(_ctx, getState());
  enterRule(_localctx, 184, qasm3Parser::RuleTimingType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(927);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__87: {
        enterOuterAlt(_localctx, 1);
        setState(922);
        match(qasm3Parser::T__87);
        break;
      }

      case qasm3Parser::T__88: {
        enterOuterAlt(_localctx, 2);
        setState(923);
        match(qasm3Parser::T__88);
        setState(925);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == qasm3Parser::Integer) {
          setState(924);
          match(qasm3Parser::Integer);
        }
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TimingBoxContext ------------------------------------------------------------------

qasm3Parser::TimingBoxContext::TimingBoxContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::TimingBoxContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::QuantumBlockContext* qasm3Parser::TimingBoxContext::quantumBlock() {
  return getRuleContext<qasm3Parser::QuantumBlockContext>(0);
}

tree::TerminalNode* qasm3Parser::TimingBoxContext::TimingLiteral() {
  return getToken(qasm3Parser::TimingLiteral, 0);
}


size_t qasm3Parser::TimingBoxContext::getRuleIndex() const {
  return qasm3Parser::RuleTimingBox;
}

void qasm3Parser::TimingBoxContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTimingBox(this);
}

void qasm3Parser::TimingBoxContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTimingBox(this);
}


antlrcpp::Any qasm3Parser::TimingBoxContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitTimingBox(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::TimingBoxContext* qasm3Parser::timingBox() {
  TimingBoxContext *_localctx = _tracker.createInstance<TimingBoxContext>(_ctx, getState());
  enterRule(_localctx, 186, qasm3Parser::RuleTimingBox);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(935);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__89: {
        enterOuterAlt(_localctx, 1);
        setState(929);
        match(qasm3Parser::T__89);
        setState(930);
        match(qasm3Parser::Identifier);
        setState(931);
        quantumBlock();
        break;
      }

      case qasm3Parser::T__90: {
        enterOuterAlt(_localctx, 2);
        setState(932);
        match(qasm3Parser::T__90);
        setState(933);
        match(qasm3Parser::TimingLiteral);
        setState(934);
        quantumBlock();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TimingTerminatorContext ------------------------------------------------------------------

qasm3Parser::TimingTerminatorContext::TimingTerminatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::TimingIdentifierContext* qasm3Parser::TimingTerminatorContext::timingIdentifier() {
  return getRuleContext<qasm3Parser::TimingIdentifierContext>(0);
}


size_t qasm3Parser::TimingTerminatorContext::getRuleIndex() const {
  return qasm3Parser::RuleTimingTerminator;
}

void qasm3Parser::TimingTerminatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTimingTerminator(this);
}

void qasm3Parser::TimingTerminatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTimingTerminator(this);
}


antlrcpp::Any qasm3Parser::TimingTerminatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitTimingTerminator(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::TimingTerminatorContext* qasm3Parser::timingTerminator() {
  TimingTerminatorContext *_localctx = _tracker.createInstance<TimingTerminatorContext>(_ctx, getState());
  enterRule(_localctx, 188, qasm3Parser::RuleTimingTerminator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(939);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__63:
      case qasm3Parser::TimingLiteral: {
        enterOuterAlt(_localctx, 1);
        setState(937);
        timingIdentifier();
        break;
      }

      case qasm3Parser::T__91: {
        enterOuterAlt(_localctx, 2);
        setState(938);
        match(qasm3Parser::T__91);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TimingIdentifierContext ------------------------------------------------------------------

qasm3Parser::TimingIdentifierContext::TimingIdentifierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::TimingIdentifierContext::TimingLiteral() {
  return getToken(qasm3Parser::TimingLiteral, 0);
}

tree::TerminalNode* qasm3Parser::TimingIdentifierContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::TimingIdentifierContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

tree::TerminalNode* qasm3Parser::TimingIdentifierContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::QuantumBlockContext* qasm3Parser::TimingIdentifierContext::quantumBlock() {
  return getRuleContext<qasm3Parser::QuantumBlockContext>(0);
}


size_t qasm3Parser::TimingIdentifierContext::getRuleIndex() const {
  return qasm3Parser::RuleTimingIdentifier;
}

void qasm3Parser::TimingIdentifierContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTimingIdentifier(this);
}

void qasm3Parser::TimingIdentifierContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTimingIdentifier(this);
}


antlrcpp::Any qasm3Parser::TimingIdentifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitTimingIdentifier(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::TimingIdentifierContext* qasm3Parser::timingIdentifier() {
  TimingIdentifierContext *_localctx = _tracker.createInstance<TimingIdentifierContext>(_ctx, getState());
  enterRule(_localctx, 190, qasm3Parser::RuleTimingIdentifier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(949);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::TimingLiteral: {
        enterOuterAlt(_localctx, 1);
        setState(941);
        match(qasm3Parser::TimingLiteral);
        break;
      }

      case qasm3Parser::T__63: {
        enterOuterAlt(_localctx, 2);
        setState(942);
        match(qasm3Parser::T__63);
        setState(943);
        match(qasm3Parser::LPAREN);
        setState(946);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case qasm3Parser::Identifier: {
            setState(944);
            match(qasm3Parser::Identifier);
            break;
          }

          case qasm3Parser::LBRACE: {
            setState(945);
            quantumBlock();
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        setState(948);
        match(qasm3Parser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TimingInstructionNameContext ------------------------------------------------------------------

qasm3Parser::TimingInstructionNameContext::TimingInstructionNameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t qasm3Parser::TimingInstructionNameContext::getRuleIndex() const {
  return qasm3Parser::RuleTimingInstructionName;
}

void qasm3Parser::TimingInstructionNameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTimingInstructionName(this);
}

void qasm3Parser::TimingInstructionNameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTimingInstructionName(this);
}


antlrcpp::Any qasm3Parser::TimingInstructionNameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitTimingInstructionName(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::TimingInstructionNameContext* qasm3Parser::timingInstructionName() {
  TimingInstructionNameContext *_localctx = _tracker.createInstance<TimingInstructionNameContext>(_ctx, getState());
  enterRule(_localctx, 192, qasm3Parser::RuleTimingInstructionName);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(951);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__92

    || _la == qasm3Parser::T__93)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TimingInstructionContext ------------------------------------------------------------------

qasm3Parser::TimingInstructionContext::TimingInstructionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::TimingInstructionNameContext* qasm3Parser::TimingInstructionContext::timingInstructionName() {
  return getRuleContext<qasm3Parser::TimingInstructionNameContext>(0);
}

qasm3Parser::DesignatorContext* qasm3Parser::TimingInstructionContext::designator() {
  return getRuleContext<qasm3Parser::DesignatorContext>(0);
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::TimingInstructionContext::indexIdentifierList() {
  return getRuleContext<qasm3Parser::IndexIdentifierListContext>(0);
}

tree::TerminalNode* qasm3Parser::TimingInstructionContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::TimingInstructionContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

qasm3Parser::ExpressionListContext* qasm3Parser::TimingInstructionContext::expressionList() {
  return getRuleContext<qasm3Parser::ExpressionListContext>(0);
}


size_t qasm3Parser::TimingInstructionContext::getRuleIndex() const {
  return qasm3Parser::RuleTimingInstruction;
}

void qasm3Parser::TimingInstructionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTimingInstruction(this);
}

void qasm3Parser::TimingInstructionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTimingInstruction(this);
}


antlrcpp::Any qasm3Parser::TimingInstructionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitTimingInstruction(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::TimingInstructionContext* qasm3Parser::timingInstruction() {
  TimingInstructionContext *_localctx = _tracker.createInstance<TimingInstructionContext>(_ctx, getState());
  enterRule(_localctx, 194, qasm3Parser::RuleTimingInstruction);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(953);
    timingInstructionName();
    setState(959);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(954);
      match(qasm3Parser::LPAREN);
      setState(956);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__15)
        | (1ULL << qasm3Parser::T__16)
        | (1ULL << qasm3Parser::T__31)
        | (1ULL << qasm3Parser::T__32)
        | (1ULL << qasm3Parser::T__51)
        | (1ULL << qasm3Parser::T__52)
        | (1ULL << qasm3Parser::T__53)
        | (1ULL << qasm3Parser::T__54)
        | (1ULL << qasm3Parser::T__55)
        | (1ULL << qasm3Parser::T__56)
        | (1ULL << qasm3Parser::T__57)
        | (1ULL << qasm3Parser::T__58)
        | (1ULL << qasm3Parser::T__59)
        | (1ULL << qasm3Parser::T__60)
        | (1ULL << qasm3Parser::T__61)
        | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
        | (1ULL << (qasm3Parser::T__87 - 64))
        | (1ULL << (qasm3Parser::T__88 - 64))
        | (1ULL << (qasm3Parser::T__91 - 64))
        | (1ULL << (qasm3Parser::LPAREN - 64))
        | (1ULL << (qasm3Parser::MINUS - 64))
        | (1ULL << (qasm3Parser::Constant - 64))
        | (1ULL << (qasm3Parser::Integer - 64))
        | (1ULL << (qasm3Parser::Identifier - 64))
        | (1ULL << (qasm3Parser::RealNumber - 64))
        | (1ULL << (qasm3Parser::TimingLiteral - 64))
        | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
        setState(955);
        expressionList();
      }
      setState(958);
      match(qasm3Parser::RPAREN);
    }
    setState(961);
    designator();
    setState(962);
    indexIdentifierList();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TimingStatementContext ------------------------------------------------------------------

qasm3Parser::TimingStatementContext::TimingStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::TimingInstructionContext* qasm3Parser::TimingStatementContext::timingInstruction() {
  return getRuleContext<qasm3Parser::TimingInstructionContext>(0);
}

tree::TerminalNode* qasm3Parser::TimingStatementContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}

qasm3Parser::TimingBoxContext* qasm3Parser::TimingStatementContext::timingBox() {
  return getRuleContext<qasm3Parser::TimingBoxContext>(0);
}


size_t qasm3Parser::TimingStatementContext::getRuleIndex() const {
  return qasm3Parser::RuleTimingStatement;
}

void qasm3Parser::TimingStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTimingStatement(this);
}

void qasm3Parser::TimingStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTimingStatement(this);
}


antlrcpp::Any qasm3Parser::TimingStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitTimingStatement(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::TimingStatementContext* qasm3Parser::timingStatement() {
  TimingStatementContext *_localctx = _tracker.createInstance<TimingStatementContext>(_ctx, getState());
  enterRule(_localctx, 196, qasm3Parser::RuleTimingStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(968);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__92:
      case qasm3Parser::T__93: {
        enterOuterAlt(_localctx, 1);
        setState(964);
        timingInstruction();
        setState(965);
        match(qasm3Parser::SEMICOLON);
        break;
      }

      case qasm3Parser::T__89:
      case qasm3Parser::T__90: {
        enterOuterAlt(_localctx, 2);
        setState(967);
        timingBox();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CalibrationContext ------------------------------------------------------------------

qasm3Parser::CalibrationContext::CalibrationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::CalibrationGrammarDeclarationContext* qasm3Parser::CalibrationContext::calibrationGrammarDeclaration() {
  return getRuleContext<qasm3Parser::CalibrationGrammarDeclarationContext>(0);
}

qasm3Parser::CalibrationDefinitionContext* qasm3Parser::CalibrationContext::calibrationDefinition() {
  return getRuleContext<qasm3Parser::CalibrationDefinitionContext>(0);
}


size_t qasm3Parser::CalibrationContext::getRuleIndex() const {
  return qasm3Parser::RuleCalibration;
}

void qasm3Parser::CalibrationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCalibration(this);
}

void qasm3Parser::CalibrationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCalibration(this);
}


antlrcpp::Any qasm3Parser::CalibrationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCalibration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::CalibrationContext* qasm3Parser::calibration() {
  CalibrationContext *_localctx = _tracker.createInstance<CalibrationContext>(_ctx, getState());
  enterRule(_localctx, 198, qasm3Parser::RuleCalibration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(972);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__94: {
        enterOuterAlt(_localctx, 1);
        setState(970);
        calibrationGrammarDeclaration();
        break;
      }

      case qasm3Parser::T__95: {
        enterOuterAlt(_localctx, 2);
        setState(971);
        calibrationDefinition();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CalibrationGrammarDeclarationContext ------------------------------------------------------------------

qasm3Parser::CalibrationGrammarDeclarationContext::CalibrationGrammarDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::CalibrationGrammarContext* qasm3Parser::CalibrationGrammarDeclarationContext::calibrationGrammar() {
  return getRuleContext<qasm3Parser::CalibrationGrammarContext>(0);
}

tree::TerminalNode* qasm3Parser::CalibrationGrammarDeclarationContext::SEMICOLON() {
  return getToken(qasm3Parser::SEMICOLON, 0);
}


size_t qasm3Parser::CalibrationGrammarDeclarationContext::getRuleIndex() const {
  return qasm3Parser::RuleCalibrationGrammarDeclaration;
}

void qasm3Parser::CalibrationGrammarDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCalibrationGrammarDeclaration(this);
}

void qasm3Parser::CalibrationGrammarDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCalibrationGrammarDeclaration(this);
}


antlrcpp::Any qasm3Parser::CalibrationGrammarDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCalibrationGrammarDeclaration(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::CalibrationGrammarDeclarationContext* qasm3Parser::calibrationGrammarDeclaration() {
  CalibrationGrammarDeclarationContext *_localctx = _tracker.createInstance<CalibrationGrammarDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 200, qasm3Parser::RuleCalibrationGrammarDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(974);
    match(qasm3Parser::T__94);
    setState(975);
    calibrationGrammar();
    setState(976);
    match(qasm3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CalibrationDefinitionContext ------------------------------------------------------------------

qasm3Parser::CalibrationDefinitionContext::CalibrationDefinitionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::CalibrationDefinitionContext::Identifier() {
  return getToken(qasm3Parser::Identifier, 0);
}

qasm3Parser::IdentifierListContext* qasm3Parser::CalibrationDefinitionContext::identifierList() {
  return getRuleContext<qasm3Parser::IdentifierListContext>(0);
}

tree::TerminalNode* qasm3Parser::CalibrationDefinitionContext::LBRACE() {
  return getToken(qasm3Parser::LBRACE, 0);
}

tree::TerminalNode* qasm3Parser::CalibrationDefinitionContext::RBRACE() {
  return getToken(qasm3Parser::RBRACE, 0);
}

tree::TerminalNode* qasm3Parser::CalibrationDefinitionContext::LPAREN() {
  return getToken(qasm3Parser::LPAREN, 0);
}

tree::TerminalNode* qasm3Parser::CalibrationDefinitionContext::RPAREN() {
  return getToken(qasm3Parser::RPAREN, 0);
}

qasm3Parser::ReturnSignatureContext* qasm3Parser::CalibrationDefinitionContext::returnSignature() {
  return getRuleContext<qasm3Parser::ReturnSignatureContext>(0);
}

qasm3Parser::CalibrationArgumentListContext* qasm3Parser::CalibrationDefinitionContext::calibrationArgumentList() {
  return getRuleContext<qasm3Parser::CalibrationArgumentListContext>(0);
}


size_t qasm3Parser::CalibrationDefinitionContext::getRuleIndex() const {
  return qasm3Parser::RuleCalibrationDefinition;
}

void qasm3Parser::CalibrationDefinitionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCalibrationDefinition(this);
}

void qasm3Parser::CalibrationDefinitionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCalibrationDefinition(this);
}


antlrcpp::Any qasm3Parser::CalibrationDefinitionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCalibrationDefinition(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::CalibrationDefinitionContext* qasm3Parser::calibrationDefinition() {
  CalibrationDefinitionContext *_localctx = _tracker.createInstance<CalibrationDefinitionContext>(_ctx, getState());
  enterRule(_localctx, 202, qasm3Parser::RuleCalibrationDefinition);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(978);
    match(qasm3Parser::T__95);
    setState(979);
    match(qasm3Parser::Identifier);
    setState(985);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(980);
      match(qasm3Parser::LPAREN);
      setState(982);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__15)
        | (1ULL << qasm3Parser::T__16)
        | (1ULL << qasm3Parser::T__31)
        | (1ULL << qasm3Parser::T__32)
        | (1ULL << qasm3Parser::T__51)
        | (1ULL << qasm3Parser::T__52)
        | (1ULL << qasm3Parser::T__53)
        | (1ULL << qasm3Parser::T__54)
        | (1ULL << qasm3Parser::T__55)
        | (1ULL << qasm3Parser::T__56)
        | (1ULL << qasm3Parser::T__57)
        | (1ULL << qasm3Parser::T__58)
        | (1ULL << qasm3Parser::T__59)
        | (1ULL << qasm3Parser::T__60)
        | (1ULL << qasm3Parser::T__61)
        | (1ULL << qasm3Parser::T__62))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 64)) & ((1ULL << (qasm3Parser::T__63 - 64))
        | (1ULL << (qasm3Parser::T__87 - 64))
        | (1ULL << (qasm3Parser::T__88 - 64))
        | (1ULL << (qasm3Parser::T__91 - 64))
        | (1ULL << (qasm3Parser::LPAREN - 64))
        | (1ULL << (qasm3Parser::MINUS - 64))
        | (1ULL << (qasm3Parser::Constant - 64))
        | (1ULL << (qasm3Parser::Integer - 64))
        | (1ULL << (qasm3Parser::Identifier - 64))
        | (1ULL << (qasm3Parser::RealNumber - 64))
        | (1ULL << (qasm3Parser::TimingLiteral - 64))
        | (1ULL << (qasm3Parser::StringLiteral - 64)))) != 0)) {
        setState(981);
        calibrationArgumentList();
      }
      setState(984);
      match(qasm3Parser::RPAREN);
    }
    setState(987);
    identifierList();
    setState(989);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::ARROW) {
      setState(988);
      returnSignature();
    }
    setState(991);
    match(qasm3Parser::LBRACE);
    setState(995);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 98, _ctx);
    while (alt != 1 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1 + 1) {
        setState(992);
        matchWildcard(); 
      }
      setState(997);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 98, _ctx);
    }
    setState(998);
    match(qasm3Parser::RBRACE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CalibrationGrammarContext ------------------------------------------------------------------

qasm3Parser::CalibrationGrammarContext::CalibrationGrammarContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasm3Parser::CalibrationGrammarContext::StringLiteral() {
  return getToken(qasm3Parser::StringLiteral, 0);
}


size_t qasm3Parser::CalibrationGrammarContext::getRuleIndex() const {
  return qasm3Parser::RuleCalibrationGrammar;
}

void qasm3Parser::CalibrationGrammarContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCalibrationGrammar(this);
}

void qasm3Parser::CalibrationGrammarContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCalibrationGrammar(this);
}


antlrcpp::Any qasm3Parser::CalibrationGrammarContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCalibrationGrammar(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::CalibrationGrammarContext* qasm3Parser::calibrationGrammar() {
  CalibrationGrammarContext *_localctx = _tracker.createInstance<CalibrationGrammarContext>(_ctx, getState());
  enterRule(_localctx, 204, qasm3Parser::RuleCalibrationGrammar);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1000);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__96

    || _la == qasm3Parser::StringLiteral)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CalibrationArgumentListContext ------------------------------------------------------------------

qasm3Parser::CalibrationArgumentListContext::CalibrationArgumentListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasm3Parser::ClassicalArgumentListContext* qasm3Parser::CalibrationArgumentListContext::classicalArgumentList() {
  return getRuleContext<qasm3Parser::ClassicalArgumentListContext>(0);
}

qasm3Parser::ExpressionListContext* qasm3Parser::CalibrationArgumentListContext::expressionList() {
  return getRuleContext<qasm3Parser::ExpressionListContext>(0);
}


size_t qasm3Parser::CalibrationArgumentListContext::getRuleIndex() const {
  return qasm3Parser::RuleCalibrationArgumentList;
}

void qasm3Parser::CalibrationArgumentListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCalibrationArgumentList(this);
}

void qasm3Parser::CalibrationArgumentListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasm3Listener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCalibrationArgumentList(this);
}


antlrcpp::Any qasm3Parser::CalibrationArgumentListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasm3Visitor*>(visitor))
    return parserVisitor->visitCalibrationArgumentList(this);
  else
    return visitor->visitChildren(this);
}

qasm3Parser::CalibrationArgumentListContext* qasm3Parser::calibrationArgumentList() {
  CalibrationArgumentListContext *_localctx = _tracker.createInstance<CalibrationArgumentListContext>(_ctx, getState());
  enterRule(_localctx, 206, qasm3Parser::RuleCalibrationArgumentList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1004);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 99, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1002);
      classicalArgumentList();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1003);
      expressionList();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool qasm3Parser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 36: return indexIdentifierSempred(dynamic_cast<IndexIdentifierContext *>(context), predicateIndex);
    case 58: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);
    case 59: return xOrExpressionSempred(dynamic_cast<XOrExpressionContext *>(context), predicateIndex);
    case 60: return bitAndExpressionSempred(dynamic_cast<BitAndExpressionContext *>(context), predicateIndex);
    case 61: return bitShiftExpressionSempred(dynamic_cast<BitShiftExpressionContext *>(context), predicateIndex);
    case 62: return additiveExpressionSempred(dynamic_cast<AdditiveExpressionContext *>(context), predicateIndex);
    case 63: return multiplicativeExpressionSempred(dynamic_cast<MultiplicativeExpressionContext *>(context), predicateIndex);
    case 65: return expressionTerminatorSempred(dynamic_cast<ExpressionTerminatorContext *>(context), predicateIndex);
    case 71: return booleanExpressionSempred(dynamic_cast<BooleanExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::indexIdentifierSempred(IndexIdentifierContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 1: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::xOrExpressionSempred(XOrExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 2: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::bitAndExpressionSempred(BitAndExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 3: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::bitShiftExpressionSempred(BitShiftExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 4: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::additiveExpressionSempred(AdditiveExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 5: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::multiplicativeExpressionSempred(MultiplicativeExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 6: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::expressionTerminatorSempred(ExpressionTerminatorContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 7: return precpred(_ctx, 2);
    case 8: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool qasm3Parser::booleanExpressionSempred(BooleanExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 9: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> qasm3Parser::_decisionToDFA;
atn::PredictionContextCache qasm3Parser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN qasm3Parser::_atn;
std::vector<uint16_t> qasm3Parser::_serializedATN;

std::vector<std::string> qasm3Parser::_ruleNames = {
  "program", "header", "version", "include", "globalStatement", "statement", 
  "compute_action_stmt", "qcor_test_statement", "quantumDeclarationStatement", 
  "classicalDeclarationStatement", "classicalAssignment", "assignmentStatement", 
  "returnSignature", "designator", "doubleDesignator", "identifierList", 
  "association", "quantumType", "quantumDeclaration", "quantumArgument", 
  "quantumArgumentList", "bitType", "singleDesignatorType", "doubleDesignatorType", 
  "noDesignatorType", "classicalType", "constantDeclaration", "singleDesignatorDeclaration", 
  "doubleDesignatorDeclaration", "noDesignatorDeclaration", "bitDeclaration", 
  "classicalDeclaration", "classicalTypeList", "classicalArgument", "classicalArgumentList", 
  "aliasStatement", "indexIdentifier", "indexIdentifierList", "indexEqualsAssignmentList", 
  "rangeDefinition", "quantumGateDefinition", "quantumGateSignature", "quantumBlock", 
  "quantumLoop", "quantumLoopBlock", "quantumStatement", "quantumInstruction", 
  "quantumPhase", "quantumMeasurement", "quantumMeasurementAssignment", 
  "quantumBarrier", "quantumGateModifier", "quantumGateCall", "quantumGateName", 
  "unaryOperator", "relationalOperator", "logicalOperator", "expressionStatement", 
  "expression", "xOrExpression", "bitAndExpression", "bitShiftExpression", 
  "additiveExpression", "multiplicativeExpression", "unaryExpression", "expressionTerminator", 
  "incrementor", "builtInCall", "builtInMath", "castOperator", "expressionList", 
  "booleanExpression", "comparsionExpression", "equalsExpression", "assignmentOperator", 
  "equalsAssignmentList", "membershipTest", "setDeclaration", "programBlock", 
  "branchingStatement", "loopSignature", "loopStatement", "cLikeLoopStatement", 
  "controlDirectiveStatement", "controlDirective", "kernelDeclaration", 
  "kernelCall", "subroutineDefinition", "returnStatement", "subroutineBlock", 
  "subroutineCall", "pragma", "timingType", "timingBox", "timingTerminator", 
  "timingIdentifier", "timingInstructionName", "timingInstruction", "timingStatement", 
  "calibration", "calibrationGrammarDeclaration", "calibrationDefinition", 
  "calibrationGrammar", "calibrationArgumentList"
};

std::vector<std::string> qasm3Parser::_literalNames = {
  "", "'OPENQASM'", "'include'", "'compute'", "'action'", "'QCOR_EXPECT_TRUE'", 
  "'qubit'", "'qreg'", "'bit'", "'creg'", "'int'", "'uint'", "'float'", 
  "'angle'", "'fixed'", "'bool'", "'int64_t'", "'double'", "'const'", "'let'", 
  "'||'", "'gate'", "'CX'", "'U'", "'gphase'", "'measure'", "'barrier'", 
  "'inv'", "'pow'", "'ctrl'", "'@'", "'reset'", "'~'", "'!'", "'>'", "'<'", 
  "'>='", "'<='", "'=='", "'!='", "'&&'", "'|'", "'^'", "'&'", "'<<'", "'>>'", 
  "'+'", "'*'", "'/'", "'%'", "'++'", "'--'", "'sin'", "'cos'", "'tan'", 
  "'arctan'", "'arccos'", "'arcsin'", "'exp'", "'ln'", "'sqrt'", "'rotl'", 
  "'rotr'", "'popcount'", "'lengthof'", "'+='", "'-='", "'*='", "'/='", 
  "'&='", "'|='", "'~='", "'^='", "'<<='", "'>>='", "'in'", "'if'", "'else'", 
  "'for'", "'while'", "'range'", "'break'", "'continue'", "'end'", "'kernel'", 
  "'def'", "'return'", "'#pragma'", "'length'", "'stretch'", "'boxas'", 
  "'boxto'", "'stretchinf'", "'delay'", "'rotary'", "'defcalgrammar'", "'defcal'", 
  "'\"openpulse\"'", "'['", "']'", "'{'", "'}'", "'('", "')'", "':'", "';'", 
  "'.'", "','", "'='", "'->'", "'-'", "'extern'"
};

std::vector<std::string> qasm3Parser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE", 
  "LPAREN", "RPAREN", "COLON", "SEMICOLON", "DOT", "COMMA", "EQUALS", "ARROW", 
  "MINUS", "EXTERN", "Constant", "Whitespace", "Newline", "Integer", "Identifier", 
  "RealNumber", "TimingLiteral", "StringLiteral", "LineComment", "BlockComment"
};

dfa::Vocabulary qasm3Parser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> qasm3Parser::_tokenNames;

qasm3Parser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  static const uint16_t serializedATNSegment0[] = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
       0x3, 0x7b, 0x3f1, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
       0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 
       0x7, 0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 
       0x4, 0xb, 0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 
       0xe, 0x9, 0xe, 0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 
       0x9, 0x11, 0x4, 0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 
       0x9, 0x14, 0x4, 0x15, 0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 
       0x9, 0x17, 0x4, 0x18, 0x9, 0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 
       0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 
       0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 0x1f, 0x9, 0x1f, 0x4, 0x20, 
       0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x4, 0x22, 0x9, 0x22, 0x4, 0x23, 
       0x9, 0x23, 0x4, 0x24, 0x9, 0x24, 0x4, 0x25, 0x9, 0x25, 0x4, 0x26, 
       0x9, 0x26, 0x4, 0x27, 0x9, 0x27, 0x4, 0x28, 0x9, 0x28, 0x4, 0x29, 
       0x9, 0x29, 0x4, 0x2a, 0x9, 0x2a, 0x4, 0x2b, 0x9, 0x2b, 0x4, 0x2c, 
       0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x4, 0x2e, 0x9, 0x2e, 0x4, 0x2f, 
       0x9, 0x2f, 0x4, 0x30, 0x9, 0x30, 0x4, 0x31, 0x9, 0x31, 0x4, 0x32, 
       0x9, 0x32, 0x4, 0x33, 0x9, 0x33, 0x4, 0x34, 0x9, 0x34, 0x4, 0x35, 
       0x9, 0x35, 0x4, 0x36, 0x9, 0x36, 0x4, 0x37, 0x9, 0x37, 0x4, 0x38, 
       0x9, 0x38, 0x4, 0x39, 0x9, 0x39, 0x4, 0x3a, 0x9, 0x3a, 0x4, 0x3b, 
       0x9, 0x3b, 0x4, 0x3c, 0x9, 0x3c, 0x4, 0x3d, 0x9, 0x3d, 0x4, 0x3e, 
       0x9, 0x3e, 0x4, 0x3f, 0x9, 0x3f, 0x4, 0x40, 0x9, 0x40, 0x4, 0x41, 
       0x9, 0x41, 0x4, 0x42, 0x9, 0x42, 0x4, 0x43, 0x9, 0x43, 0x4, 0x44, 
       0x9, 0x44, 0x4, 0x45, 0x9, 0x45, 0x4, 0x46, 0x9, 0x46, 0x4, 0x47, 
       0x9, 0x47, 0x4, 0x48, 0x9, 0x48, 0x4, 0x49, 0x9, 0x49, 0x4, 0x4a, 
       0x9, 0x4a, 0x4, 0x4b, 0x9, 0x4b, 0x4, 0x4c, 0x9, 0x4c, 0x4, 0x4d, 
       0x9, 0x4d, 0x4, 0x4e, 0x9, 0x4e, 0x4, 0x4f, 0x9, 0x4f, 0x4, 0x50, 
       0x9, 0x50, 0x4, 0x51, 0x9, 0x51, 0x4, 0x52, 0x9, 0x52, 0x4, 0x53, 
       0x9, 0x53, 0x4, 0x54, 0x9, 0x54, 0x4, 0x55, 0x9, 0x55, 0x4, 0x56, 
       0x9, 0x56, 0x4, 0x57, 0x9, 0x57, 0x4, 0x58, 0x9, 0x58, 0x4, 0x59, 
       0x9, 0x59, 0x4, 0x5a, 0x9, 0x5a, 0x4, 0x5b, 0x9, 0x5b, 0x4, 0x5c, 
       0x9, 0x5c, 0x4, 0x5d, 0x9, 0x5d, 0x4, 0x5e, 0x9, 0x5e, 0x4, 0x5f, 
       0x9, 0x5f, 0x4, 0x60, 0x9, 0x60, 0x4, 0x61, 0x9, 0x61, 0x4, 0x62, 
       0x9, 0x62, 0x4, 0x63, 0x9, 0x63, 0x4, 0x64, 0x9, 0x64, 0x4, 0x65, 
       0x9, 0x65, 0x4, 0x66, 0x9, 0x66, 0x4, 0x67, 0x9, 0x67, 0x4, 0x68, 
       0x9, 0x68, 0x4, 0x69, 0x9, 0x69, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x7, 
       0x2, 0xd6, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0xd9, 0xb, 0x2, 0x3, 0x3, 
       0x5, 0x3, 0xdc, 0xa, 0x3, 0x3, 0x3, 0x7, 0x3, 0xdf, 0xa, 0x3, 0xc, 
       0x3, 0xe, 0x3, 0xe2, 0xb, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
       0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0xf2, 0xa, 0x6, 
       0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
       0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0xff, 0xa, 
       0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 
       0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 
       0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x111, 0xa, 0xb, 0x3, 
       0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x5, 0xc, 
       0x119, 0xa, 0xc, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x11d, 0xa, 0xd, 0x3, 
       0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 
       0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
       0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 0x7, 0x11, 0x130, 0xa, 0x11, 
       0xc, 0x11, 0xe, 0x11, 0x133, 0xb, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
       0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 0x3, 
       0x14, 0x3, 0x14, 0x5, 0x14, 0x13f, 0xa, 0x14, 0x3, 0x14, 0x3, 0x14, 
       0x5, 0x14, 0x143, 0xa, 0x14, 0x3, 0x14, 0x5, 0x14, 0x146, 0xa, 0x14, 
       0x3, 0x15, 0x3, 0x15, 0x5, 0x15, 0x14a, 0xa, 0x15, 0x3, 0x15, 0x3, 
       0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x7, 0x16, 0x151, 0xa, 0x16, 
       0xc, 0x16, 0xe, 0x16, 0x154, 0xb, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 
       0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 
       0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 
       0x1a, 0x164, 0xa, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 
       0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x5, 0x1b, 
       0x16f, 0xa, 0x1b, 0x5, 0x1b, 0x171, 0xa, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 
       0x3, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 
       0x17a, 0xa, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x5, 
       0x1e, 0x180, 0xa, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x5, 0x1f, 
       0x185, 0xa, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x5, 0x20, 0x18a, 
       0xa, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x5, 0x21, 
       0x190, 0xa, 0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x7, 0x22, 0x195, 
       0xa, 0x22, 0xc, 0x22, 0xe, 0x22, 0x198, 0xb, 0x22, 0x3, 0x22, 0x3, 
       0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 
       0x24, 0x7, 0x24, 0x1a2, 0xa, 0x24, 0xc, 0x24, 0xe, 0x24, 0x1a5, 0xb, 
       0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 
       0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 
       0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x5, 0x26, 0x1b7, 
       0xa, 0x26, 0x5, 0x26, 0x1b9, 0xa, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 
       0x26, 0x7, 0x26, 0x1be, 0xa, 0x26, 0xc, 0x26, 0xe, 0x26, 0x1c1, 0xb, 
       0x26, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 0x1c6, 0xa, 0x27, 
       0xc, 0x27, 0xe, 0x27, 0x1c9, 0xb, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 
       0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x7, 0x28, 0x1d1, 0xa, 0x28, 
       0xc, 0x28, 0xe, 0x28, 0x1d4, 0xb, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 
       0x28, 0x3, 0x29, 0x3, 0x29, 0x5, 0x29, 0x1db, 0xa, 0x29, 0x3, 0x29, 
       0x3, 0x29, 0x5, 0x29, 0x1df, 0xa, 0x29, 0x3, 0x29, 0x3, 0x29, 0x5, 
       0x29, 0x1e3, 0xa, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 
       0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 
       0x1ee, 0xa, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 0x1f1, 0xa, 0x2b, 0x3, 0x2b, 
       0x3, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x7, 0x2c, 
       0x1f9, 0xa, 0x2c, 0xc, 0x2c, 0xe, 0x2c, 0x1fc, 0xb, 0x2c, 0x3, 0x2c, 
       0x3, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 
       0x3, 0x2e, 0x7, 0x2e, 0x206, 0xa, 0x2e, 0xc, 0x2e, 0xe, 0x2e, 0x209, 
       0xb, 0x2e, 0x3, 0x2e, 0x5, 0x2e, 0x20c, 0xa, 0x2e, 0x3, 0x2f, 0x3, 
       0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x5, 0x2f, 0x212, 0xa, 0x2f, 0x3, 0x30, 
       0x3, 0x30, 0x3, 0x30, 0x3, 0x30, 0x5, 0x30, 0x218, 0xa, 0x30, 0x3, 
       0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x32, 0x3, 
       0x32, 0x3, 0x32, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x5, 0x33, 0x225, 
       0xa, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x5, 0x33, 
       0x22b, 0xa, 0x33, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x35, 0x3, 
       0x35, 0x3, 0x35, 0x3, 0x35, 0x3, 0x35, 0x3, 0x35, 0x3, 0x35, 0x5, 
       0x35, 0x237, 0xa, 0x35, 0x3, 0x35, 0x3, 0x35, 0x3, 0x36, 0x3, 0x36, 
       0x3, 0x36, 0x5, 0x36, 0x23e, 0xa, 0x36, 0x3, 0x36, 0x5, 0x36, 0x241, 
       0xa, 0x36, 0x3, 0x36, 0x3, 0x36, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
       0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x5, 0x37, 0x24c, 0xa, 
       0x37, 0x3, 0x38, 0x3, 0x38, 0x3, 0x39, 0x3, 0x39, 0x3, 0x3a, 0x3, 
       0x3a, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3c, 0x3, 0x3c, 0x3, 
       0x3c, 0x3, 0x3c, 0x5, 0x3c, 0x25b, 0xa, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 
       0x3, 0x3c, 0x7, 0x3c, 0x260, 0xa, 0x3c, 0xc, 0x3c, 0xe, 0x3c, 0x263, 
       0xb, 0x3c, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 
       0x3, 0x3d, 0x7, 0x3d, 0x26b, 0xa, 0x3d, 0xc, 0x3d, 0xe, 0x3d, 0x26e, 
       0xb, 0x3d, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 
       0x3, 0x3e, 0x7, 0x3e, 0x276, 0xa, 0x3e, 0xc, 0x3e, 0xe, 0x3e, 0x279, 
       0xb, 0x3e, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 
       0x3, 0x3f, 0x7, 0x3f, 0x281, 0xa, 0x3f, 0xc, 0x3f, 0xe, 0x3f, 0x284, 
       0xb, 0x3f, 0x3, 0x40, 0x3, 0x40, 0x3, 0x40, 0x3, 0x40, 0x3, 0x40, 
       0x3, 0x40, 0x7, 0x40, 0x28c, 0xa, 0x40, 0xc, 0x40, 0xe, 0x40, 0x28f, 
       0xb, 0x40, 0x3, 0x41, 0x3, 0x41, 0x3, 0x41, 0x5, 0x41, 0x294, 0xa, 
       0x41, 0x3, 0x41, 0x3, 0x41, 0x3, 0x41, 0x3, 0x41, 0x5, 0x41, 0x29a, 
       0xa, 0x41, 0x7, 0x41, 0x29c, 0xa, 0x41, 0xc, 0x41, 0xe, 0x41, 0x29f, 
       0xb, 0x41, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x43, 0x3, 0x43, 
       0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 
       0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 
       0x3, 0x43, 0x3, 0x43, 0x5, 0x43, 0x2b4, 0xa, 0x43, 0x3, 0x43, 0x3, 
       0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x3, 0x43, 0x7, 
       0x43, 0x2bd, 0xa, 0x43, 0xc, 0x43, 0xe, 0x43, 0x2c0, 0xb, 0x43, 0x3, 
       0x44, 0x3, 0x44, 0x3, 0x45, 0x3, 0x45, 0x5, 0x45, 0x2c6, 0xa, 0x45, 
       0x3, 0x45, 0x3, 0x45, 0x3, 0x45, 0x3, 0x45, 0x3, 0x46, 0x3, 0x46, 
       0x3, 0x47, 0x3, 0x47, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x7, 0x48, 
       0x2d3, 0xa, 0x48, 0xc, 0x48, 0xe, 0x48, 0x2d6, 0xb, 0x48, 0x3, 0x48, 
       0x3, 0x48, 0x3, 0x49, 0x3, 0x49, 0x3, 0x49, 0x5, 0x49, 0x2dd, 0xa, 
       0x49, 0x3, 0x49, 0x3, 0x49, 0x3, 0x49, 0x3, 0x49, 0x7, 0x49, 0x2e3, 
       0xa, 0x49, 0xc, 0x49, 0xe, 0x49, 0x2e6, 0xb, 0x49, 0x3, 0x4a, 0x3, 
       0x4a, 0x3, 0x4a, 0x3, 0x4a, 0x3, 0x4a, 0x5, 0x4a, 0x2ed, 0xa, 0x4a, 
       0x3, 0x4b, 0x3, 0x4b, 0x3, 0x4b, 0x3, 0x4c, 0x3, 0x4c, 0x3, 0x4d, 
       0x3, 0x4d, 0x3, 0x4d, 0x3, 0x4d, 0x7, 0x4d, 0x2f8, 0xa, 0x4d, 0xc, 
       0x4d, 0xe, 0x4d, 0x2fb, 0xb, 0x4d, 0x3, 0x4d, 0x3, 0x4d, 0x3, 0x4d, 
       0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4f, 0x3, 0x4f, 
       0x3, 0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x5, 0x4f, 0x30a, 0xa, 
       0x4f, 0x3, 0x50, 0x3, 0x50, 0x3, 0x50, 0x7, 0x50, 0x30f, 0xa, 0x50, 
       0xc, 0x50, 0xe, 0x50, 0x312, 0xb, 0x50, 0x3, 0x50, 0x5, 0x50, 0x315, 
       0xa, 0x50, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 
       0x3, 0x51, 0x3, 0x51, 0x5, 0x51, 0x31e, 0xa, 0x51, 0x3, 0x52, 0x3, 
       0x52, 0x3, 0x52, 0x3, 0x52, 0x3, 0x52, 0x3, 0x52, 0x3, 0x52, 0x5, 
       0x52, 0x327, 0xa, 0x52, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x54, 
       0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 
       0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 
       0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 
       0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 
       0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 0x5, 0x54, 0x348, 0xa, 0x54, 0x3, 
       0x55, 0x3, 0x55, 0x3, 0x55, 0x3, 0x56, 0x3, 0x56, 0x3, 0x57, 0x3, 
       0x57, 0x3, 0x57, 0x3, 0x57, 0x5, 0x57, 0x353, 0xa, 0x57, 0x3, 0x57, 
       0x5, 0x57, 0x356, 0xa, 0x57, 0x3, 0x57, 0x5, 0x57, 0x359, 0xa, 0x57, 
       0x3, 0x57, 0x5, 0x57, 0x35c, 0xa, 0x57, 0x3, 0x57, 0x3, 0x57, 0x3, 
       0x58, 0x3, 0x58, 0x3, 0x58, 0x5, 0x58, 0x363, 0xa, 0x58, 0x3, 0x58, 
       0x3, 0x58, 0x3, 0x59, 0x3, 0x59, 0x3, 0x59, 0x3, 0x59, 0x5, 0x59, 
       0x36b, 0xa, 0x59, 0x3, 0x59, 0x5, 0x59, 0x36e, 0xa, 0x59, 0x3, 0x59, 
       0x5, 0x59, 0x371, 0xa, 0x59, 0x3, 0x59, 0x5, 0x59, 0x374, 0xa, 0x59, 
       0x3, 0x59, 0x3, 0x59, 0x3, 0x5a, 0x3, 0x5a, 0x3, 0x5a, 0x3, 0x5b, 
       0x3, 0x5b, 0x7, 0x5b, 0x37d, 0xa, 0x5b, 0xc, 0x5b, 0xe, 0x5b, 0x380, 
       0xb, 0x5b, 0x3, 0x5b, 0x5, 0x5b, 0x383, 0xa, 0x5b, 0x3, 0x5b, 0x3, 
       0x5b, 0x3, 0x5b, 0x5, 0x5b, 0x388, 0xa, 0x5b, 0x3, 0x5c, 0x3, 0x5c, 
       0x3, 0x5c, 0x5, 0x5c, 0x38d, 0xa, 0x5c, 0x3, 0x5c, 0x3, 0x5c, 0x3, 
       0x5c, 0x3, 0x5c, 0x3, 0x5d, 0x3, 0x5d, 0x3, 0x5d, 0x7, 0x5d, 0x396, 
       0xa, 0x5d, 0xc, 0x5d, 0xe, 0x5d, 0x399, 0xb, 0x5d, 0x3, 0x5d, 0x3, 
       0x5d, 0x3, 0x5e, 0x3, 0x5e, 0x3, 0x5e, 0x5, 0x5e, 0x3a0, 0xa, 0x5e, 
       0x5, 0x5e, 0x3a2, 0xa, 0x5e, 0x3, 0x5f, 0x3, 0x5f, 0x3, 0x5f, 0x3, 
       0x5f, 0x3, 0x5f, 0x3, 0x5f, 0x5, 0x5f, 0x3aa, 0xa, 0x5f, 0x3, 0x60, 
       0x3, 0x60, 0x5, 0x60, 0x3ae, 0xa, 0x60, 0x3, 0x61, 0x3, 0x61, 0x3, 
       0x61, 0x3, 0x61, 0x3, 0x61, 0x5, 0x61, 0x3b5, 0xa, 0x61, 0x3, 0x61, 
       0x5, 0x61, 0x3b8, 0xa, 0x61, 0x3, 0x62, 0x3, 0x62, 0x3, 0x63, 0x3, 
       0x63, 0x3, 0x63, 0x5, 0x63, 0x3bf, 0xa, 0x63, 0x3, 0x63, 0x5, 0x63, 
       0x3c2, 0xa, 0x63, 0x3, 0x63, 0x3, 0x63, 0x3, 0x63, 0x3, 0x64, 0x3, 
       0x64, 0x3, 0x64, 0x3, 0x64, 0x5, 0x64, 0x3cb, 0xa, 0x64, 0x3, 0x65, 
       0x3, 0x65, 0x5, 0x65, 0x3cf, 0xa, 0x65, 0x3, 0x66, 0x3, 0x66, 0x3, 
       0x66, 0x3, 0x66, 0x3, 0x67, 0x3, 0x67, 0x3, 0x67, 0x3, 0x67, 0x5, 
       0x67, 0x3d9, 0xa, 0x67, 0x3, 0x67, 0x5, 0x67, 0x3dc, 0xa, 0x67, 0x3, 
       0x67, 0x3, 0x67, 0x5, 0x67, 0x3e0, 0xa, 0x67, 0x3, 0x67, 0x3, 0x67, 
       0x7, 0x67, 0x3e4, 0xa, 0x67, 0xc, 0x67, 0xe, 0x67, 0x3e7, 0xb, 0x67, 
       0x3, 0x67, 0x3, 0x67, 0x3, 0x68, 0x3, 0x68, 0x3, 0x69, 0x3, 0x69, 
       0x5, 0x69, 0x3ef, 0xa, 0x69, 0x3, 0x69, 0x3, 0x3e5, 0xb, 0x4a, 0x76, 
       0x78, 0x7a, 0x7c, 0x7e, 0x80, 0x84, 0x90, 0x6a, 0x2, 0x4, 0x6, 0x8, 
       0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 
       0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 
       0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 
       0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x62, 
       0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 
       0x7a, 0x7c, 0x7e, 0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 
       0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e, 0xa0, 0xa2, 0xa4, 
       0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 
       0xbc, 0xbe, 0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 
       0x2, 0x13, 0x4, 0x2, 0x75, 0x75, 0x77, 0x77, 0x3, 0x2, 0x8, 0x9, 
       0x3, 0x2, 0xa, 0xb, 0x3, 0x2, 0xc, 0xf, 0x4, 0x2, 0x18, 0x19, 0x76, 
       0x76, 0x3, 0x2, 0x22, 0x23, 0x3, 0x2, 0x24, 0x29, 0x4, 0x2, 0x16, 
       0x16, 0x2a, 0x2a, 0x3, 0x2, 0x2e, 0x2f, 0x4, 0x2, 0x30, 0x30, 0x70, 
       0x70, 0x3, 0x2, 0x31, 0x33, 0x3, 0x2, 0x34, 0x35, 0x3, 0x2, 0x36, 
       0x42, 0x4, 0x2, 0x43, 0x4c, 0x6e, 0x6e, 0x3, 0x2, 0x53, 0x55, 0x3, 
       0x2, 0x5f, 0x60, 0x4, 0x2, 0x63, 0x63, 0x79, 0x79, 0x2, 0x413, 0x2, 
       0xd2, 0x3, 0x2, 0x2, 0x2, 0x4, 0xdb, 0x3, 0x2, 0x2, 0x2, 0x6, 0xe3, 
       0x3, 0x2, 0x2, 0x2, 0x8, 0xe7, 0x3, 0x2, 0x2, 0x2, 0xa, 0xf1, 0x3, 
       0x2, 0x2, 0x2, 0xc, 0xfe, 0x3, 0x2, 0x2, 0x2, 0xe, 0x100, 0x3, 0x2, 
       0x2, 0x2, 0x10, 0x105, 0x3, 0x2, 0x2, 0x2, 0x12, 0x10b, 0x3, 0x2, 
       0x2, 0x2, 0x14, 0x110, 0x3, 0x2, 0x2, 0x2, 0x16, 0x114, 0x3, 0x2, 
       0x2, 0x2, 0x18, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x1a, 0x120, 0x3, 0x2, 
       0x2, 0x2, 0x1c, 0x123, 0x3, 0x2, 0x2, 0x2, 0x1e, 0x127, 0x3, 0x2, 
       0x2, 0x2, 0x20, 0x131, 0x3, 0x2, 0x2, 0x2, 0x22, 0x136, 0x3, 0x2, 
       0x2, 0x2, 0x24, 0x139, 0x3, 0x2, 0x2, 0x2, 0x26, 0x145, 0x3, 0x2, 
       0x2, 0x2, 0x28, 0x147, 0x3, 0x2, 0x2, 0x2, 0x2a, 0x152, 0x3, 0x2, 
       0x2, 0x2, 0x2c, 0x157, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x159, 0x3, 0x2, 
       0x2, 0x2, 0x30, 0x15b, 0x3, 0x2, 0x2, 0x2, 0x32, 0x163, 0x3, 0x2, 
       0x2, 0x2, 0x34, 0x170, 0x3, 0x2, 0x2, 0x2, 0x36, 0x172, 0x3, 0x2, 
       0x2, 0x2, 0x38, 0x175, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x17b, 0x3, 0x2, 
       0x2, 0x2, 0x3c, 0x181, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x186, 0x3, 0x2, 
       0x2, 0x2, 0x40, 0x18f, 0x3, 0x2, 0x2, 0x2, 0x42, 0x196, 0x3, 0x2, 
       0x2, 0x2, 0x44, 0x19b, 0x3, 0x2, 0x2, 0x2, 0x46, 0x1a3, 0x3, 0x2, 
       0x2, 0x2, 0x48, 0x1a8, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x1b8, 0x3, 0x2, 
       0x2, 0x2, 0x4c, 0x1c7, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x1d2, 0x3, 0x2, 
       0x2, 0x2, 0x50, 0x1d8, 0x3, 0x2, 0x2, 0x2, 0x52, 0x1e6, 0x3, 0x2, 
       0x2, 0x2, 0x54, 0x1ea, 0x3, 0x2, 0x2, 0x2, 0x56, 0x1f4, 0x3, 0x2, 
       0x2, 0x2, 0x58, 0x1ff, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x20b, 0x3, 0x2, 
       0x2, 0x2, 0x5c, 0x211, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x217, 0x3, 0x2, 
       0x2, 0x2, 0x60, 0x219, 0x3, 0x2, 0x2, 0x2, 0x62, 0x21e, 0x3, 0x2, 
       0x2, 0x2, 0x64, 0x22a, 0x3, 0x2, 0x2, 0x2, 0x66, 0x22c, 0x3, 0x2, 
       0x2, 0x2, 0x68, 0x236, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x23a, 0x3, 0x2, 
       0x2, 0x2, 0x6c, 0x24b, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x24d, 0x3, 0x2, 
       0x2, 0x2, 0x70, 0x24f, 0x3, 0x2, 0x2, 0x2, 0x72, 0x251, 0x3, 0x2, 
       0x2, 0x2, 0x74, 0x253, 0x3, 0x2, 0x2, 0x2, 0x76, 0x25a, 0x3, 0x2, 
       0x2, 0x2, 0x78, 0x264, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x26f, 0x3, 0x2, 
       0x2, 0x2, 0x7c, 0x27a, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x285, 0x3, 0x2, 
       0x2, 0x2, 0x80, 0x293, 0x3, 0x2, 0x2, 0x2, 0x82, 0x2a0, 0x3, 0x2, 
       0x2, 0x2, 0x84, 0x2b3, 0x3, 0x2, 0x2, 0x2, 0x86, 0x2c1, 0x3, 0x2, 
       0x2, 0x2, 0x88, 0x2c5, 0x3, 0x2, 0x2, 0x2, 0x8a, 0x2cb, 0x3, 0x2, 
       0x2, 0x2, 0x8c, 0x2cd, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x2d4, 0x3, 0x2, 
       0x2, 0x2, 0x90, 0x2dc, 0x3, 0x2, 0x2, 0x2, 0x92, 0x2ec, 0x3, 0x2, 
       0x2, 0x2, 0x94, 0x2ee, 0x3, 0x2, 0x2, 0x2, 0x96, 0x2f1, 0x3, 0x2, 
       0x2, 0x2, 0x98, 0x2f9, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x2ff, 0x3, 0x2, 
       0x2, 0x2, 0x9c, 0x309, 0x3, 0x2, 0x2, 0x2, 0x9e, 0x314, 0x3, 0x2, 
       0x2, 0x2, 0xa0, 0x316, 0x3, 0x2, 0x2, 0x2, 0xa2, 0x326, 0x3, 0x2, 
       0x2, 0x2, 0xa4, 0x328, 0x3, 0x2, 0x2, 0x2, 0xa6, 0x347, 0x3, 0x2, 
       0x2, 0x2, 0xa8, 0x349, 0x3, 0x2, 0x2, 0x2, 0xaa, 0x34c, 0x3, 0x2, 
       0x2, 0x2, 0xac, 0x34e, 0x3, 0x2, 0x2, 0x2, 0xae, 0x35f, 0x3, 0x2, 
       0x2, 0x2, 0xb0, 0x366, 0x3, 0x2, 0x2, 0x2, 0xb2, 0x377, 0x3, 0x2, 
       0x2, 0x2, 0xb4, 0x387, 0x3, 0x2, 0x2, 0x2, 0xb6, 0x389, 0x3, 0x2, 
       0x2, 0x2, 0xb8, 0x392, 0x3, 0x2, 0x2, 0x2, 0xba, 0x3a1, 0x3, 0x2, 
       0x2, 0x2, 0xbc, 0x3a9, 0x3, 0x2, 0x2, 0x2, 0xbe, 0x3ad, 0x3, 0x2, 
       0x2, 0x2, 0xc0, 0x3b7, 0x3, 0x2, 0x2, 0x2, 0xc2, 0x3b9, 0x3, 0x2, 
       0x2, 0x2, 0xc4, 0x3bb, 0x3, 0x2, 0x2, 0x2, 0xc6, 0x3ca, 0x3, 0x2, 
       0x2, 0x2, 0xc8, 0x3ce, 0x3, 0x2, 0x2, 0x2, 0xca, 0x3d0, 0x3, 0x2, 
       0x2, 0x2, 0xcc, 0x3d4, 0x3, 0x2, 0x2, 0x2, 0xce, 0x3ea, 0x3, 0x2, 
       0x2, 0x2, 0xd0, 0x3ee, 0x3, 0x2, 0x2, 0x2, 0xd2, 0xd7, 0x5, 0x4, 
       0x3, 0x2, 0xd3, 0xd6, 0x5, 0xa, 0x6, 0x2, 0xd4, 0xd6, 0x5, 0xc, 0x7, 
       0x2, 0xd5, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd4, 0x3, 0x2, 0x2, 0x2, 
       0xd6, 0xd9, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xd5, 0x3, 0x2, 0x2, 0x2, 0xd7, 
       0xd8, 0x3, 0x2, 0x2, 0x2, 0xd8, 0x3, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xd7, 
       0x3, 0x2, 0x2, 0x2, 0xda, 0xdc, 0x5, 0x6, 0x4, 0x2, 0xdb, 0xda, 0x3, 
       0x2, 0x2, 0x2, 0xdb, 0xdc, 0x3, 0x2, 0x2, 0x2, 0xdc, 0xe0, 0x3, 0x2, 
       0x2, 0x2, 0xdd, 0xdf, 0x5, 0x8, 0x5, 0x2, 0xde, 0xdd, 0x3, 0x2, 0x2, 
       0x2, 0xdf, 0xe2, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xde, 0x3, 0x2, 0x2, 0x2, 
       0xe0, 0xe1, 0x3, 0x2, 0x2, 0x2, 0xe1, 0x5, 0x3, 0x2, 0x2, 0x2, 0xe2, 
       0xe0, 0x3, 0x2, 0x2, 0x2, 0xe3, 0xe4, 0x7, 0x3, 0x2, 0x2, 0xe4, 0xe5, 
       0x9, 0x2, 0x2, 0x2, 0xe5, 0xe6, 0x7, 0x6b, 0x2, 0x2, 0xe6, 0x7, 0x3, 
       0x2, 0x2, 0x2, 0xe7, 0xe8, 0x7, 0x4, 0x2, 0x2, 0xe8, 0xe9, 0x7, 0x79, 
       0x2, 0x2, 0xe9, 0xea, 0x7, 0x6b, 0x2, 0x2, 0xea, 0x9, 0x3, 0x2, 0x2, 
       0x2, 0xeb, 0xf2, 0x5, 0xb0, 0x59, 0x2, 0xec, 0xf2, 0x5, 0xac, 0x57, 
       0x2, 0xed, 0xf2, 0x5, 0x52, 0x2a, 0x2, 0xee, 0xf2, 0x5, 0xc8, 0x65, 
       0x2, 0xef, 0xf2, 0x5, 0x12, 0xa, 0x2, 0xf0, 0xf2, 0x5, 0xb8, 0x5d, 
       0x2, 0xf1, 0xeb, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xec, 0x3, 0x2, 0x2, 0x2, 
       0xf1, 0xed, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xee, 0x3, 0x2, 0x2, 0x2, 0xf1, 
       0xef, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xf0, 0x3, 0x2, 0x2, 0x2, 0xf2, 0xb, 
       0x3, 0x2, 0x2, 0x2, 0xf3, 0xff, 0x5, 0x74, 0x3b, 0x2, 0xf4, 0xff, 
       0x5, 0x18, 0xd, 0x2, 0xf5, 0xff, 0x5, 0x14, 0xb, 0x2, 0xf6, 0xff, 
       0x5, 0xa0, 0x51, 0x2, 0xf7, 0xff, 0x5, 0xa4, 0x53, 0x2, 0xf8, 0xff, 
       0x5, 0xa8, 0x55, 0x2, 0xf9, 0xff, 0x5, 0x48, 0x25, 0x2, 0xfa, 0xff, 
       0x5, 0x5c, 0x2f, 0x2, 0xfb, 0xff, 0x5, 0xb2, 0x5a, 0x2, 0xfc, 0xff, 
       0x5, 0x10, 0x9, 0x2, 0xfd, 0xff, 0x5, 0xe, 0x8, 0x2, 0xfe, 0xf3, 
       0x3, 0x2, 0x2, 0x2, 0xfe, 0xf4, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xf5, 0x3, 
       0x2, 0x2, 0x2, 0xfe, 0xf6, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xf7, 0x3, 0x2, 
       0x2, 0x2, 0xfe, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xf9, 0x3, 0x2, 0x2, 
       0x2, 0xfe, 0xfa, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xfb, 0x3, 0x2, 0x2, 0x2, 
       0xfe, 0xfc, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xfd, 0x3, 0x2, 0x2, 0x2, 0xff, 
       0xd, 0x3, 0x2, 0x2, 0x2, 0x100, 0x101, 0x7, 0x5, 0x2, 0x2, 0x101, 
       0x102, 0x5, 0x9e, 0x50, 0x2, 0x102, 0x103, 0x7, 0x6, 0x2, 0x2, 0x103, 
       0x104, 0x5, 0x9e, 0x50, 0x2, 0x104, 0xf, 0x3, 0x2, 0x2, 0x2, 0x105, 
       0x106, 0x7, 0x7, 0x2, 0x2, 0x106, 0x107, 0x7, 0x68, 0x2, 0x2, 0x107, 
       0x108, 0x5, 0x90, 0x49, 0x2, 0x108, 0x109, 0x7, 0x69, 0x2, 0x2, 0x109, 
       0x10a, 0x7, 0x6b, 0x2, 0x2, 0x10a, 0x11, 0x3, 0x2, 0x2, 0x2, 0x10b, 
       0x10c, 0x5, 0x26, 0x14, 0x2, 0x10c, 0x10d, 0x7, 0x6b, 0x2, 0x2, 0x10d, 
       0x13, 0x3, 0x2, 0x2, 0x2, 0x10e, 0x111, 0x5, 0x40, 0x21, 0x2, 0x10f, 
       0x111, 0x5, 0x36, 0x1c, 0x2, 0x110, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x110, 
       0x10f, 0x3, 0x2, 0x2, 0x2, 0x111, 0x112, 0x3, 0x2, 0x2, 0x2, 0x112, 
       0x113, 0x7, 0x6b, 0x2, 0x2, 0x113, 0x15, 0x3, 0x2, 0x2, 0x2, 0x114, 
       0x115, 0x5, 0x4a, 0x26, 0x2, 0x115, 0x118, 0x5, 0x96, 0x4c, 0x2, 
       0x116, 0x119, 0x5, 0x76, 0x3c, 0x2, 0x117, 0x119, 0x5, 0x4a, 0x26, 
       0x2, 0x118, 0x116, 0x3, 0x2, 0x2, 0x2, 0x118, 0x117, 0x3, 0x2, 0x2, 
       0x2, 0x119, 0x17, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x11d, 0x5, 0x16, 0xc, 
       0x2, 0x11b, 0x11d, 0x5, 0x64, 0x33, 0x2, 0x11c, 0x11a, 0x3, 0x2, 
       0x2, 0x2, 0x11c, 0x11b, 0x3, 0x2, 0x2, 0x2, 0x11d, 0x11e, 0x3, 0x2, 
       0x2, 0x2, 0x11e, 0x11f, 0x7, 0x6b, 0x2, 0x2, 0x11f, 0x19, 0x3, 0x2, 
       0x2, 0x2, 0x120, 0x121, 0x7, 0x6f, 0x2, 0x2, 0x121, 0x122, 0x5, 0x34, 
       0x1b, 0x2, 0x122, 0x1b, 0x3, 0x2, 0x2, 0x2, 0x123, 0x124, 0x7, 0x64, 
       0x2, 0x2, 0x124, 0x125, 0x5, 0x76, 0x3c, 0x2, 0x125, 0x126, 0x7, 
       0x65, 0x2, 0x2, 0x126, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x127, 0x128, 0x7, 
       0x64, 0x2, 0x2, 0x128, 0x129, 0x5, 0x76, 0x3c, 0x2, 0x129, 0x12a, 
       0x7, 0x6d, 0x2, 0x2, 0x12a, 0x12b, 0x5, 0x76, 0x3c, 0x2, 0x12b, 0x12c, 
       0x7, 0x65, 0x2, 0x2, 0x12c, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x12e, 
       0x7, 0x76, 0x2, 0x2, 0x12e, 0x130, 0x7, 0x6d, 0x2, 0x2, 0x12f, 0x12d, 
       0x3, 0x2, 0x2, 0x2, 0x130, 0x133, 0x3, 0x2, 0x2, 0x2, 0x131, 0x12f, 
       0x3, 0x2, 0x2, 0x2, 0x131, 0x132, 0x3, 0x2, 0x2, 0x2, 0x132, 0x134, 
       0x3, 0x2, 0x2, 0x2, 0x133, 0x131, 0x3, 0x2, 0x2, 0x2, 0x134, 0x135, 
       0x7, 0x76, 0x2, 0x2, 0x135, 0x21, 0x3, 0x2, 0x2, 0x2, 0x136, 0x137, 
       0x7, 0x6a, 0x2, 0x2, 0x137, 0x138, 0x7, 0x76, 0x2, 0x2, 0x138, 0x23, 
       0x3, 0x2, 0x2, 0x2, 0x139, 0x13a, 0x9, 0x3, 0x2, 0x2, 0x13a, 0x25, 
       0x3, 0x2, 0x2, 0x2, 0x13b, 0x13c, 0x7, 0x9, 0x2, 0x2, 0x13c, 0x13e, 
       0x7, 0x76, 0x2, 0x2, 0x13d, 0x13f, 0x5, 0x1c, 0xf, 0x2, 0x13e, 0x13d, 
       0x3, 0x2, 0x2, 0x2, 0x13e, 0x13f, 0x3, 0x2, 0x2, 0x2, 0x13f, 0x146, 
       0x3, 0x2, 0x2, 0x2, 0x140, 0x142, 0x7, 0x8, 0x2, 0x2, 0x141, 0x143, 
       0x5, 0x1c, 0xf, 0x2, 0x142, 0x141, 0x3, 0x2, 0x2, 0x2, 0x142, 0x143, 
       0x3, 0x2, 0x2, 0x2, 0x143, 0x144, 0x3, 0x2, 0x2, 0x2, 0x144, 0x146, 
       0x7, 0x76, 0x2, 0x2, 0x145, 0x13b, 0x3, 0x2, 0x2, 0x2, 0x145, 0x140, 
       0x3, 0x2, 0x2, 0x2, 0x146, 0x27, 0x3, 0x2, 0x2, 0x2, 0x147, 0x149, 
       0x5, 0x24, 0x13, 0x2, 0x148, 0x14a, 0x5, 0x1c, 0xf, 0x2, 0x149, 0x148, 
       0x3, 0x2, 0x2, 0x2, 0x149, 0x14a, 0x3, 0x2, 0x2, 0x2, 0x14a, 0x14b, 
       0x3, 0x2, 0x2, 0x2, 0x14b, 0x14c, 0x5, 0x22, 0x12, 0x2, 0x14c, 0x29, 
       0x3, 0x2, 0x2, 0x2, 0x14d, 0x14e, 0x5, 0x28, 0x15, 0x2, 0x14e, 0x14f, 
       0x7, 0x6d, 0x2, 0x2, 0x14f, 0x151, 0x3, 0x2, 0x2, 0x2, 0x150, 0x14d, 
       0x3, 0x2, 0x2, 0x2, 0x151, 0x154, 0x3, 0x2, 0x2, 0x2, 0x152, 0x150, 
       0x3, 0x2, 0x2, 0x2, 0x152, 0x153, 0x3, 0x2, 0x2, 0x2, 0x153, 0x155, 
       0x3, 0x2, 0x2, 0x2, 0x154, 0x152, 0x3, 0x2, 0x2, 0x2, 0x155, 0x156, 
       0x5, 0x28, 0x15, 0x2, 0x156, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x157, 0x158, 
       0x9, 0x4, 0x2, 0x2, 0x158, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x159, 0x15a, 
       0x9, 0x5, 0x2, 0x2, 0x15a, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x15b, 0x15c, 
       0x7, 0x10, 0x2, 0x2, 0x15c, 0x31, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x164, 
       0x7, 0x11, 0x2, 0x2, 0x15e, 0x164, 0x5, 0xba, 0x5e, 0x2, 0x15f, 0x164, 
       0x7, 0xc, 0x2, 0x2, 0x160, 0x164, 0x7, 0x12, 0x2, 0x2, 0x161, 0x164, 
       0x7, 0xe, 0x2, 0x2, 0x162, 0x164, 0x7, 0x13, 0x2, 0x2, 0x163, 0x15d, 
       0x3, 0x2, 0x2, 0x2, 0x163, 0x15e, 0x3, 0x2, 0x2, 0x2, 0x163, 0x15f, 
       0x3, 0x2, 0x2, 0x2, 0x163, 0x160, 0x3, 0x2, 0x2, 0x2, 0x163, 0x161, 
       0x3, 0x2, 0x2, 0x2, 0x163, 0x162, 0x3, 0x2, 0x2, 0x2, 0x164, 0x33, 
       0x3, 0x2, 0x2, 0x2, 0x165, 0x166, 0x5, 0x2e, 0x18, 0x2, 0x166, 0x167, 
       0x5, 0x1c, 0xf, 0x2, 0x167, 0x171, 0x3, 0x2, 0x2, 0x2, 0x168, 0x169, 
       0x5, 0x30, 0x19, 0x2, 0x169, 0x16a, 0x5, 0x1e, 0x10, 0x2, 0x16a, 
       0x171, 0x3, 0x2, 0x2, 0x2, 0x16b, 0x171, 0x5, 0x32, 0x1a, 0x2, 0x16c, 
       0x16e, 0x5, 0x2c, 0x17, 0x2, 0x16d, 0x16f, 0x5, 0x1c, 0xf, 0x2, 0x16e, 
       0x16d, 0x3, 0x2, 0x2, 0x2, 0x16e, 0x16f, 0x3, 0x2, 0x2, 0x2, 0x16f, 
       0x171, 0x3, 0x2, 0x2, 0x2, 0x170, 0x165, 0x3, 0x2, 0x2, 0x2, 0x170, 
       0x168, 0x3, 0x2, 0x2, 0x2, 0x170, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x170, 
       0x16c, 0x3, 0x2, 0x2, 0x2, 0x171, 0x35, 0x3, 0x2, 0x2, 0x2, 0x172, 
       0x173, 0x7, 0x14, 0x2, 0x2, 0x173, 0x174, 0x5, 0x98, 0x4d, 0x2, 0x174, 
       0x37, 0x3, 0x2, 0x2, 0x2, 0x175, 0x176, 0x5, 0x2e, 0x18, 0x2, 0x176, 
       0x179, 0x5, 0x1c, 0xf, 0x2, 0x177, 0x17a, 0x5, 0x20, 0x11, 0x2, 0x178, 
       0x17a, 0x5, 0x98, 0x4d, 0x2, 0x179, 0x177, 0x3, 0x2, 0x2, 0x2, 0x179, 
       0x178, 0x3, 0x2, 0x2, 0x2, 0x17a, 0x39, 0x3, 0x2, 0x2, 0x2, 0x17b, 
       0x17c, 0x5, 0x30, 0x19, 0x2, 0x17c, 0x17f, 0x5, 0x1e, 0x10, 0x2, 
       0x17d, 0x180, 0x5, 0x20, 0x11, 0x2, 0x17e, 0x180, 0x5, 0x98, 0x4d, 
       0x2, 0x17f, 0x17d, 0x3, 0x2, 0x2, 0x2, 0x17f, 0x17e, 0x3, 0x2, 0x2, 
       0x2, 0x180, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x181, 0x184, 0x5, 0x32, 0x1a, 
       0x2, 0x182, 0x185, 0x5, 0x20, 0x11, 0x2, 0x183, 0x185, 0x5, 0x98, 
       0x4d, 0x2, 0x184, 0x182, 0x3, 0x2, 0x2, 0x2, 0x184, 0x183, 0x3, 0x2, 
       0x2, 0x2, 0x185, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x186, 0x189, 0x5, 0x2c, 
       0x17, 0x2, 0x187, 0x18a, 0x5, 0x4c, 0x27, 0x2, 0x188, 0x18a, 0x5, 
       0x4e, 0x28, 0x2, 0x189, 0x187, 0x3, 0x2, 0x2, 0x2, 0x189, 0x188, 
       0x3, 0x2, 0x2, 0x2, 0x18a, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x18b, 0x190, 
       0x5, 0x38, 0x1d, 0x2, 0x18c, 0x190, 0x5, 0x3a, 0x1e, 0x2, 0x18d, 
       0x190, 0x5, 0x3c, 0x1f, 0x2, 0x18e, 0x190, 0x5, 0x3e, 0x20, 0x2, 
       0x18f, 0x18b, 0x3, 0x2, 0x2, 0x2, 0x18f, 0x18c, 0x3, 0x2, 0x2, 0x2, 
       0x18f, 0x18d, 0x3, 0x2, 0x2, 0x2, 0x18f, 0x18e, 0x3, 0x2, 0x2, 0x2, 
       0x190, 0x41, 0x3, 0x2, 0x2, 0x2, 0x191, 0x192, 0x5, 0x34, 0x1b, 0x2, 
       0x192, 0x193, 0x7, 0x6d, 0x2, 0x2, 0x193, 0x195, 0x3, 0x2, 0x2, 0x2, 
       0x194, 0x191, 0x3, 0x2, 0x2, 0x2, 0x195, 0x198, 0x3, 0x2, 0x2, 0x2, 
       0x196, 0x194, 0x3, 0x2, 0x2, 0x2, 0x196, 0x197, 0x3, 0x2, 0x2, 0x2, 
       0x197, 0x199, 0x3, 0x2, 0x2, 0x2, 0x198, 0x196, 0x3, 0x2, 0x2, 0x2, 
       0x199, 0x19a, 0x5, 0x34, 0x1b, 0x2, 0x19a, 0x43, 0x3, 0x2, 0x2, 0x2, 
       0x19b, 0x19c, 0x5, 0x34, 0x1b, 0x2, 0x19c, 0x19d, 0x5, 0x22, 0x12, 
       0x2, 0x19d, 0x45, 0x3, 0x2, 0x2, 0x2, 0x19e, 0x19f, 0x5, 0x44, 0x23, 
       0x2, 0x19f, 0x1a0, 0x7, 0x6d, 0x2, 0x2, 0x1a0, 0x1a2, 0x3, 0x2, 0x2, 
       0x2, 0x1a1, 0x19e, 0x3, 0x2, 0x2, 0x2, 0x1a2, 0x1a5, 0x3, 0x2, 0x2, 
       0x2, 0x1a3, 0x1a1, 0x3, 0x2, 0x2, 0x2, 0x1a3, 0x1a4, 0x3, 0x2, 0x2, 
       0x2, 0x1a4, 0x1a6, 0x3, 0x2, 0x2, 0x2, 0x1a5, 0x1a3, 0x3, 0x2, 0x2, 
       0x2, 0x1a6, 0x1a7, 0x5, 0x44, 0x23, 0x2, 0x1a7, 0x47, 0x3, 0x2, 0x2, 
       0x2, 0x1a8, 0x1a9, 0x7, 0x15, 0x2, 0x2, 0x1a9, 0x1aa, 0x7, 0x76, 
       0x2, 0x2, 0x1aa, 0x1ab, 0x7, 0x6e, 0x2, 0x2, 0x1ab, 0x1ac, 0x5, 0x4a, 
       0x26, 0x2, 0x1ac, 0x1ad, 0x7, 0x6b, 0x2, 0x2, 0x1ad, 0x49, 0x3, 0x2, 
       0x2, 0x2, 0x1ae, 0x1af, 0x8, 0x26, 0x1, 0x2, 0x1af, 0x1b0, 0x7, 0x76, 
       0x2, 0x2, 0x1b0, 0x1b9, 0x5, 0x50, 0x29, 0x2, 0x1b1, 0x1b6, 0x7, 
       0x76, 0x2, 0x2, 0x1b2, 0x1b3, 0x7, 0x64, 0x2, 0x2, 0x1b3, 0x1b4, 
       0x5, 0x8e, 0x48, 0x2, 0x1b4, 0x1b5, 0x7, 0x65, 0x2, 0x2, 0x1b5, 0x1b7, 
       0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b2, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b7, 
       0x3, 0x2, 0x2, 0x2, 0x1b7, 0x1b9, 0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1ae, 
       0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b9, 0x1bf, 
       0x3, 0x2, 0x2, 0x2, 0x1ba, 0x1bb, 0xc, 0x3, 0x2, 0x2, 0x1bb, 0x1bc, 
       0x7, 0x16, 0x2, 0x2, 0x1bc, 0x1be, 0x5, 0x4a, 0x26, 0x4, 0x1bd, 0x1ba, 
       0x3, 0x2, 0x2, 0x2, 0x1be, 0x1c1, 0x3, 0x2, 0x2, 0x2, 0x1bf, 0x1bd, 
       0x3, 0x2, 0x2, 0x2, 0x1bf, 0x1c0, 0x3, 0x2, 0x2, 0x2, 0x1c0, 0x4b, 
       0x3, 0x2, 0x2, 0x2, 0x1c1, 0x1bf, 0x3, 0x2, 0x2, 0x2, 0x1c2, 0x1c3, 
       0x5, 0x4a, 0x26, 0x2, 0x1c3, 0x1c4, 0x7, 0x6d, 0x2, 0x2, 0x1c4, 0x1c6, 
       0x3, 0x2, 0x2, 0x2, 0x1c5, 0x1c2, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c9, 
       0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1c5, 0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1c8, 
       0x3, 0x2, 0x2, 0x2, 0x1c8, 0x1ca, 0x3, 0x2, 0x2, 0x2, 0x1c9, 0x1c7, 
       0x3, 0x2, 0x2, 0x2, 0x1ca, 0x1cb, 0x5, 0x4a, 0x26, 0x2, 0x1cb, 0x4d, 
       0x3, 0x2, 0x2, 0x2, 0x1cc, 0x1cd, 0x5, 0x4a, 0x26, 0x2, 0x1cd, 0x1ce, 
       0x5, 0x94, 0x4b, 0x2, 0x1ce, 0x1cf, 0x7, 0x6d, 0x2, 0x2, 0x1cf, 0x1d1, 
       0x3, 0x2, 0x2, 0x2, 0x1d0, 0x1cc, 0x3, 0x2, 0x2, 0x2, 0x1d1, 0x1d4, 
       0x3, 0x2, 0x2, 0x2, 0x1d2, 0x1d0, 0x3, 0x2, 0x2, 0x2, 0x1d2, 0x1d3, 
       0x3, 0x2, 0x2, 0x2, 0x1d3, 0x1d5, 0x3, 0x2, 0x2, 0x2, 0x1d4, 0x1d2, 
       0x3, 0x2, 0x2, 0x2, 0x1d5, 0x1d6, 0x5, 0x4a, 0x26, 0x2, 0x1d6, 0x1d7, 
       0x5, 0x94, 0x4b, 0x2, 0x1d7, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x1d8, 0x1da, 
       0x7, 0x64, 0x2, 0x2, 0x1d9, 0x1db, 0x5, 0x76, 0x3c, 0x2, 0x1da, 0x1d9, 
       0x3, 0x2, 0x2, 0x2, 0x1da, 0x1db, 0x3, 0x2, 0x2, 0x2, 0x1db, 0x1dc, 
       0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1de, 0x7, 0x6a, 0x2, 0x2, 0x1dd, 0x1df, 
       0x5, 0x76, 0x3c, 0x2, 0x1de, 0x1dd, 0x3, 0x2, 0x2, 0x2, 0x1de, 0x1df, 
       0x3, 0x2, 0x2, 0x2, 0x1df, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x1e0, 0x1e1, 
       0x7, 0x6a, 0x2, 0x2, 0x1e1, 0x1e3, 0x5, 0x76, 0x3c, 0x2, 0x1e2, 0x1e0, 
       0x3, 0x2, 0x2, 0x2, 0x1e2, 0x1e3, 0x3, 0x2, 0x2, 0x2, 0x1e3, 0x1e4, 
       0x3, 0x2, 0x2, 0x2, 0x1e4, 0x1e5, 0x7, 0x65, 0x2, 0x2, 0x1e5, 0x51, 
       0x3, 0x2, 0x2, 0x2, 0x1e6, 0x1e7, 0x7, 0x17, 0x2, 0x2, 0x1e7, 0x1e8, 
       0x5, 0x54, 0x2b, 0x2, 0x1e8, 0x1e9, 0x5, 0x56, 0x2c, 0x2, 0x1e9, 
       0x53, 0x3, 0x2, 0x2, 0x2, 0x1ea, 0x1f0, 0x9, 0x6, 0x2, 0x2, 0x1eb, 
       0x1ed, 0x7, 0x68, 0x2, 0x2, 0x1ec, 0x1ee, 0x5, 0x20, 0x11, 0x2, 0x1ed, 
       0x1ec, 0x3, 0x2, 0x2, 0x2, 0x1ed, 0x1ee, 0x3, 0x2, 0x2, 0x2, 0x1ee, 
       0x1ef, 0x3, 0x2, 0x2, 0x2, 0x1ef, 0x1f1, 0x7, 0x69, 0x2, 0x2, 0x1f0, 
       0x1eb, 0x3, 0x2, 0x2, 0x2, 0x1f0, 0x1f1, 0x3, 0x2, 0x2, 0x2, 0x1f1, 
       0x1f2, 0x3, 0x2, 0x2, 0x2, 0x1f2, 0x1f3, 0x5, 0x20, 0x11, 0x2, 0x1f3, 
       0x55, 0x3, 0x2, 0x2, 0x2, 0x1f4, 0x1fa, 0x7, 0x66, 0x2, 0x2, 0x1f5, 
       0x1f9, 0x5, 0xe, 0x8, 0x2, 0x1f6, 0x1f9, 0x5, 0x5c, 0x2f, 0x2, 0x1f7, 
       0x1f9, 0x5, 0x58, 0x2d, 0x2, 0x1f8, 0x1f5, 0x3, 0x2, 0x2, 0x2, 0x1f8, 
       0x1f6, 0x3, 0x2, 0x2, 0x2, 0x1f8, 0x1f7, 0x3, 0x2, 0x2, 0x2, 0x1f9, 
       0x1fc, 0x3, 0x2, 0x2, 0x2, 0x1fa, 0x1f8, 0x3, 0x2, 0x2, 0x2, 0x1fa, 
       0x1fb, 0x3, 0x2, 0x2, 0x2, 0x1fb, 0x1fd, 0x3, 0x2, 0x2, 0x2, 0x1fc, 
       0x1fa, 0x3, 0x2, 0x2, 0x2, 0x1fd, 0x1fe, 0x7, 0x67, 0x2, 0x2, 0x1fe, 
       0x57, 0x3, 0x2, 0x2, 0x2, 0x1ff, 0x200, 0x5, 0xa2, 0x52, 0x2, 0x200, 
       0x201, 0x5, 0x5a, 0x2e, 0x2, 0x201, 0x59, 0x3, 0x2, 0x2, 0x2, 0x202, 
       0x20c, 0x5, 0x5c, 0x2f, 0x2, 0x203, 0x207, 0x7, 0x66, 0x2, 0x2, 0x204, 
       0x206, 0x5, 0x5c, 0x2f, 0x2, 0x205, 0x204, 0x3, 0x2, 0x2, 0x2, 0x206, 
       0x209, 0x3, 0x2, 0x2, 0x2, 0x207, 0x205, 0x3, 0x2, 0x2, 0x2, 0x207, 
       0x208, 0x3, 0x2, 0x2, 0x2, 0x208, 0x20a, 0x3, 0x2, 0x2, 0x2, 0x209, 
       0x207, 0x3, 0x2, 0x2, 0x2, 0x20a, 0x20c, 0x7, 0x67, 0x2, 0x2, 0x20b, 
       0x202, 0x3, 0x2, 0x2, 0x2, 0x20b, 0x203, 0x3, 0x2, 0x2, 0x2, 0x20c, 
       0x5b, 0x3, 0x2, 0x2, 0x2, 0x20d, 0x20e, 0x5, 0x5e, 0x30, 0x2, 0x20e, 
       0x20f, 0x7, 0x6b, 0x2, 0x2, 0x20f, 0x212, 0x3, 0x2, 0x2, 0x2, 0x210, 
       0x212, 0x5, 0xc6, 0x64, 0x2, 0x211, 0x20d, 0x3, 0x2, 0x2, 0x2, 0x211, 
       0x210, 0x3, 0x2, 0x2, 0x2, 0x212, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x213, 
       0x218, 0x5, 0x6a, 0x36, 0x2, 0x214, 0x218, 0x5, 0x60, 0x31, 0x2, 
       0x215, 0x218, 0x5, 0x62, 0x32, 0x2, 0x216, 0x218, 0x5, 0x66, 0x34, 
       0x2, 0x217, 0x213, 0x3, 0x2, 0x2, 0x2, 0x217, 0x214, 0x3, 0x2, 0x2, 
       0x2, 0x217, 0x215, 0x3, 0x2, 0x2, 0x2, 0x217, 0x216, 0x3, 0x2, 0x2, 
       0x2, 0x218, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x219, 0x21a, 0x7, 0x1a, 0x2, 
       0x2, 0x21a, 0x21b, 0x7, 0x68, 0x2, 0x2, 0x21b, 0x21c, 0x7, 0x76, 
       0x2, 0x2, 0x21c, 0x21d, 0x7, 0x69, 0x2, 0x2, 0x21d, 0x61, 0x3, 0x2, 
       0x2, 0x2, 0x21e, 0x21f, 0x7, 0x1b, 0x2, 0x2, 0x21f, 0x220, 0x5, 0x4c, 
       0x27, 0x2, 0x220, 0x63, 0x3, 0x2, 0x2, 0x2, 0x221, 0x224, 0x5, 0x62, 
       0x32, 0x2, 0x222, 0x223, 0x7, 0x6f, 0x2, 0x2, 0x223, 0x225, 0x5, 
       0x4c, 0x27, 0x2, 0x224, 0x222, 0x3, 0x2, 0x2, 0x2, 0x224, 0x225, 
       0x3, 0x2, 0x2, 0x2, 0x225, 0x22b, 0x3, 0x2, 0x2, 0x2, 0x226, 0x227, 
       0x5, 0x4c, 0x27, 0x2, 0x227, 0x228, 0x7, 0x6e, 0x2, 0x2, 0x228, 0x229, 
       0x5, 0x62, 0x32, 0x2, 0x229, 0x22b, 0x3, 0x2, 0x2, 0x2, 0x22a, 0x221, 
       0x3, 0x2, 0x2, 0x2, 0x22a, 0x226, 0x3, 0x2, 0x2, 0x2, 0x22b, 0x65, 
       0x3, 0x2, 0x2, 0x2, 0x22c, 0x22d, 0x7, 0x1c, 0x2, 0x2, 0x22d, 0x22e, 
       0x5, 0x4c, 0x27, 0x2, 0x22e, 0x67, 0x3, 0x2, 0x2, 0x2, 0x22f, 0x237, 
       0x7, 0x1d, 0x2, 0x2, 0x230, 0x231, 0x7, 0x1e, 0x2, 0x2, 0x231, 0x232, 
       0x7, 0x68, 0x2, 0x2, 0x232, 0x233, 0x5, 0x76, 0x3c, 0x2, 0x233, 0x234, 
       0x7, 0x69, 0x2, 0x2, 0x234, 0x237, 0x3, 0x2, 0x2, 0x2, 0x235, 0x237, 
       0x7, 0x1f, 0x2, 0x2, 0x236, 0x22f, 0x3, 0x2, 0x2, 0x2, 0x236, 0x230, 
       0x3, 0x2, 0x2, 0x2, 0x236, 0x235, 0x3, 0x2, 0x2, 0x2, 0x237, 0x238, 
       0x3, 0x2, 0x2, 0x2, 0x238, 0x239, 0x7, 0x20, 0x2, 0x2, 0x239, 0x69, 
       0x3, 0x2, 0x2, 0x2, 0x23a, 0x240, 0x5, 0x6c, 0x37, 0x2, 0x23b, 0x23d, 
       0x7, 0x68, 0x2, 0x2, 0x23c, 0x23e, 0x5, 0x8e, 0x48, 0x2, 0x23d, 0x23c, 
       0x3, 0x2, 0x2, 0x2, 0x23d, 0x23e, 0x3, 0x2, 0x2, 0x2, 0x23e, 0x23f, 
       0x3, 0x2, 0x2, 0x2, 0x23f, 0x241, 0x7, 0x69, 0x2, 0x2, 0x240, 0x23b, 
       0x3, 0x2, 0x2, 0x2, 0x240, 0x241, 0x3, 0x2, 0x2, 0x2, 0x241, 0x242, 
       0x3, 0x2, 0x2, 0x2, 0x242, 0x243, 0x5, 0x4c, 0x27, 0x2, 0x243, 0x6b, 
       0x3, 0x2, 0x2, 0x2, 0x244, 0x24c, 0x7, 0x18, 0x2, 0x2, 0x245, 0x24c, 
       0x7, 0x19, 0x2, 0x2, 0x246, 0x24c, 0x7, 0x21, 0x2, 0x2, 0x247, 0x24c, 
       0x7, 0x76, 0x2, 0x2, 0x248, 0x249, 0x5, 0x68, 0x35, 0x2, 0x249, 0x24a, 
       0x5, 0x6c, 0x37, 0x2, 0x24a, 0x24c, 0x3, 0x2, 0x2, 0x2, 0x24b, 0x244, 
       0x3, 0x2, 0x2, 0x2, 0x24b, 0x245, 0x3, 0x2, 0x2, 0x2, 0x24b, 0x246, 
       0x3, 0x2, 0x2, 0x2, 0x24b, 0x247, 0x3, 0x2, 0x2, 0x2, 0x24b, 0x248, 
       0x3, 0x2, 0x2, 0x2, 0x24c, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x24d, 0x24e, 
       0x9, 0x7, 0x2, 0x2, 0x24e, 0x6f, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x250, 
       0x9, 0x8, 0x2, 0x2, 0x250, 0x71, 0x3, 0x2, 0x2, 0x2, 0x251, 0x252, 
       0x9, 0x9, 0x2, 0x2, 0x252, 0x73, 0x3, 0x2, 0x2, 0x2, 0x253, 0x254, 
       0x5, 0x76, 0x3c, 0x2, 0x254, 0x255, 0x7, 0x6b, 0x2, 0x2, 0x255, 0x75, 
       0x3, 0x2, 0x2, 0x2, 0x256, 0x257, 0x8, 0x3c, 0x1, 0x2, 0x257, 0x25b, 
       0x5, 0x84, 0x43, 0x2, 0x258, 0x25b, 0x5, 0x82, 0x42, 0x2, 0x259, 
       0x25b, 0x5, 0x78, 0x3d, 0x2, 0x25a, 0x256, 0x3, 0x2, 0x2, 0x2, 0x25a, 
       0x258, 0x3, 0x2, 0x2, 0x2, 0x25a, 0x259, 0x3, 0x2, 0x2, 0x2, 0x25b, 
       0x261, 0x3, 0x2, 0x2, 0x2, 0x25c, 0x25d, 0xc, 0x3, 0x2, 0x2, 0x25d, 
       0x25e, 0x7, 0x2b, 0x2, 0x2, 0x25e, 0x260, 0x5, 0x78, 0x3d, 0x2, 0x25f, 
       0x25c, 0x3, 0x2, 0x2, 0x2, 0x260, 0x263, 0x3, 0x2, 0x2, 0x2, 0x261, 
       0x25f, 0x3, 0x2, 0x2, 0x2, 0x261, 0x262, 0x3, 0x2, 0x2, 0x2, 0x262, 
       0x77, 0x3, 0x2, 0x2, 0x2, 0x263, 0x261, 0x3, 0x2, 0x2, 0x2, 0x264, 
       0x265, 0x8, 0x3d, 0x1, 0x2, 0x265, 0x266, 0x5, 0x7a, 0x3e, 0x2, 0x266, 
       0x26c, 0x3, 0x2, 0x2, 0x2, 0x267, 0x268, 0xc, 0x3, 0x2, 0x2, 0x268, 
       0x269, 0x7, 0x2c, 0x2, 0x2, 0x269, 0x26b, 0x5, 0x7a, 0x3e, 0x2, 0x26a, 
       0x267, 0x3, 0x2, 0x2, 0x2, 0x26b, 0x26e, 0x3, 0x2, 0x2, 0x2, 0x26c, 
       0x26a, 0x3, 0x2, 0x2, 0x2, 0x26c, 0x26d, 0x3, 0x2, 0x2, 0x2, 0x26d, 
       0x79, 0x3, 0x2, 0x2, 0x2, 0x26e, 0x26c, 0x3, 0x2, 0x2, 0x2, 0x26f, 
       0x270, 0x8, 0x3e, 0x1, 0x2, 0x270, 0x271, 0x5, 0x7c, 0x3f, 0x2, 0x271, 
       0x277, 0x3, 0x2, 0x2, 0x2, 0x272, 0x273, 0xc, 0x3, 0x2, 0x2, 0x273, 
       0x274, 0x7, 0x2d, 0x2, 0x2, 0x274, 0x276, 0x5, 0x7c, 0x3f, 0x2, 0x275, 
       0x272, 0x3, 0x2, 0x2, 0x2, 0x276, 0x279, 0x3, 0x2, 0x2, 0x2, 0x277, 
       0x275, 0x3, 0x2, 0x2, 0x2, 0x277, 0x278, 0x3, 0x2, 0x2, 0x2, 0x278, 
       0x7b, 0x3, 0x2, 0x2, 0x2, 0x279, 0x277, 0x3, 0x2, 0x2, 0x2, 0x27a, 
       0x27b, 0x8, 0x3f, 0x1, 0x2, 0x27b, 0x27c, 0x5, 0x7e, 0x40, 0x2, 0x27c, 
       0x282, 0x3, 0x2, 0x2, 0x2, 0x27d, 0x27e, 0xc, 0x3, 0x2, 0x2, 0x27e, 
       0x27f, 0x9, 0xa, 0x2, 0x2, 0x27f, 0x281, 0x5, 0x7e, 0x40, 0x2, 0x280, 
       0x27d, 0x3, 0x2, 0x2, 0x2, 0x281, 0x284, 0x3, 0x2, 0x2, 0x2, 0x282, 
       0x280, 0x3, 0x2, 0x2, 0x2, 0x282, 0x283, 0x3, 0x2, 0x2, 0x2, 0x283, 
       0x7d, 0x3, 0x2, 0x2, 0x2, 0x284, 0x282, 0x3, 0x2, 0x2, 0x2, 0x285, 
       0x286, 0x8, 0x40, 0x1, 0x2, 0x286, 0x287, 0x5, 0x80, 0x41, 0x2, 0x287, 
       0x28d, 0x3, 0x2, 0x2, 0x2, 0x288, 0x289, 0xc, 0x3, 0x2, 0x2, 0x289, 
       0x28a, 0x9, 0xb, 0x2, 0x2, 0x28a, 0x28c, 0x5, 0x80, 0x41, 0x2, 0x28b, 
       0x288, 0x3, 0x2, 0x2, 0x2, 0x28c, 0x28f, 0x3, 0x2, 0x2, 0x2, 0x28d, 
       0x28b, 0x3, 0x2, 0x2, 0x2, 0x28d, 0x28e, 0x3, 0x2, 0x2, 0x2, 0x28e, 
       0x7f, 0x3, 0x2, 0x2, 0x2, 0x28f, 0x28d, 0x3, 0x2, 0x2, 0x2, 0x290, 
       0x291, 0x8, 0x41, 0x1, 0x2, 0x291, 0x294, 0x5, 0x84, 0x43, 0x2, 0x292, 
       0x294, 0x5, 0x82, 0x42, 0x2, 0x293, 0x290, 0x3, 0x2, 0x2, 0x2, 0x293, 
       0x292, 0x3, 0x2, 0x2, 0x2, 0x294, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x295, 
       0x296, 0xc, 0x3, 0x2, 0x2, 0x296, 0x299, 0x9, 0xc, 0x2, 0x2, 0x297, 
       0x29a, 0x5, 0x84, 0x43, 0x2, 0x298, 0x29a, 0x5, 0x82, 0x42, 0x2, 
       0x299, 0x297, 0x3, 0x2, 0x2, 0x2, 0x299, 0x298, 0x3, 0x2, 0x2, 0x2, 
       0x29a, 0x29c, 0x3, 0x2, 0x2, 0x2, 0x29b, 0x295, 0x3, 0x2, 0x2, 0x2, 
       0x29c, 0x29f, 0x3, 0x2, 0x2, 0x2, 0x29d, 0x29b, 0x3, 0x2, 0x2, 0x2, 
       0x29d, 0x29e, 0x3, 0x2, 0x2, 0x2, 0x29e, 0x81, 0x3, 0x2, 0x2, 0x2, 
       0x29f, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x2a0, 0x2a1, 0x5, 0x6e, 0x38, 
       0x2, 0x2a1, 0x2a2, 0x5, 0x84, 0x43, 0x2, 0x2a2, 0x83, 0x3, 0x2, 0x2, 
       0x2, 0x2a3, 0x2a4, 0x8, 0x43, 0x1, 0x2, 0x2a4, 0x2b4, 0x7, 0x72, 
       0x2, 0x2, 0x2a5, 0x2b4, 0x7, 0x75, 0x2, 0x2, 0x2a6, 0x2b4, 0x7, 0x77, 
       0x2, 0x2, 0x2a7, 0x2b4, 0x7, 0x76, 0x2, 0x2, 0x2a8, 0x2b4, 0x7, 0x79, 
       0x2, 0x2, 0x2a9, 0x2b4, 0x5, 0x88, 0x45, 0x2, 0x2aa, 0x2b4, 0x5, 
       0xae, 0x58, 0x2, 0x2ab, 0x2b4, 0x5, 0xb6, 0x5c, 0x2, 0x2ac, 0x2b4, 
       0x5, 0xbe, 0x60, 0x2, 0x2ad, 0x2ae, 0x7, 0x70, 0x2, 0x2, 0x2ae, 0x2b4, 
       0x5, 0x84, 0x43, 0x6, 0x2af, 0x2b0, 0x7, 0x68, 0x2, 0x2, 0x2b0, 0x2b1, 
       0x5, 0x76, 0x3c, 0x2, 0x2b1, 0x2b2, 0x7, 0x69, 0x2, 0x2, 0x2b2, 0x2b4, 
       0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2a3, 0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2a5, 
       0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2a6, 0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2a7, 
       0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2a8, 0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2a9, 
       0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2aa, 0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2ab, 
       0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2ac, 0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2ad, 
       0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2af, 0x3, 0x2, 0x2, 0x2, 0x2b4, 0x2be, 
       0x3, 0x2, 0x2, 0x2, 0x2b5, 0x2b6, 0xc, 0x4, 0x2, 0x2, 0x2b6, 0x2b7, 
       0x7, 0x64, 0x2, 0x2, 0x2b7, 0x2b8, 0x5, 0x76, 0x3c, 0x2, 0x2b8, 0x2b9, 
       0x7, 0x65, 0x2, 0x2, 0x2b9, 0x2bd, 0x3, 0x2, 0x2, 0x2, 0x2ba, 0x2bb, 
       0xc, 0x3, 0x2, 0x2, 0x2bb, 0x2bd, 0x5, 0x86, 0x44, 0x2, 0x2bc, 0x2b5, 
       0x3, 0x2, 0x2, 0x2, 0x2bc, 0x2ba, 0x3, 0x2, 0x2, 0x2, 0x2bd, 0x2c0, 
       0x3, 0x2, 0x2, 0x2, 0x2be, 0x2bc, 0x3, 0x2, 0x2, 0x2, 0x2be, 0x2bf, 
       0x3, 0x2, 0x2, 0x2, 0x2bf, 0x85, 0x3, 0x2, 0x2, 0x2, 0x2c0, 0x2be, 
       0x3, 0x2, 0x2, 0x2, 0x2c1, 0x2c2, 0x9, 0xd, 0x2, 0x2, 0x2c2, 0x87, 
       0x3, 0x2, 0x2, 0x2, 0x2c3, 0x2c6, 0x5, 0x8a, 0x46, 0x2, 0x2c4, 0x2c6, 
       0x5, 0x8c, 0x47, 0x2, 0x2c5, 0x2c3, 0x3, 0x2, 0x2, 0x2, 0x2c5, 0x2c4, 
       0x3, 0x2, 0x2, 0x2, 0x2c6, 0x2c7, 0x3, 0x2, 0x2, 0x2, 0x2c7, 0x2c8, 
       0x7, 0x68, 0x2, 0x2, 0x2c8, 0x2c9, 0x5, 0x8e, 0x48, 0x2, 0x2c9, 0x2ca, 
       0x7, 0x69, 0x2, 0x2, 0x2ca, 0x89, 0x3, 0x2, 0x2, 0x2, 0x2cb, 0x2cc, 
       0x9, 0xe, 0x2, 0x2, 0x2cc, 0x8b, 0x3, 0x2, 0x2, 0x2, 0x2cd, 0x2ce, 
       0x5, 0x34, 0x1b, 0x2, 0x2ce, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x2cf, 0x2d0, 
       0x5, 0x76, 0x3c, 0x2, 0x2d0, 0x2d1, 0x7, 0x6d, 0x2, 0x2, 0x2d1, 0x2d3, 
       0x3, 0x2, 0x2, 0x2, 0x2d2, 0x2cf, 0x3, 0x2, 0x2, 0x2, 0x2d3, 0x2d6, 
       0x3, 0x2, 0x2, 0x2, 0x2d4, 0x2d2, 0x3, 0x2, 0x2, 0x2, 0x2d4, 0x2d5, 
       0x3, 0x2, 0x2, 0x2, 0x2d5, 0x2d7, 0x3, 0x2, 0x2, 0x2, 0x2d6, 0x2d4, 
       0x3, 0x2, 0x2, 0x2, 0x2d7, 0x2d8, 0x5, 0x76, 0x3c, 0x2, 0x2d8, 0x8f, 
       0x3, 0x2, 0x2, 0x2, 0x2d9, 0x2da, 0x8, 0x49, 0x1, 0x2, 0x2da, 0x2dd, 
       0x5, 0x9a, 0x4e, 0x2, 0x2db, 0x2dd, 0x5, 0x92, 0x4a, 0x2, 0x2dc, 
       0x2d9, 0x3, 0x2, 0x2, 0x2, 0x2dc, 0x2db, 0x3, 0x2, 0x2, 0x2, 0x2dd, 
       0x2e4, 0x3, 0x2, 0x2, 0x2, 0x2de, 0x2df, 0xc, 0x3, 0x2, 0x2, 0x2df, 
       0x2e0, 0x5, 0x72, 0x3a, 0x2, 0x2e0, 0x2e1, 0x5, 0x92, 0x4a, 0x2, 
       0x2e1, 0x2e3, 0x3, 0x2, 0x2, 0x2, 0x2e2, 0x2de, 0x3, 0x2, 0x2, 0x2, 
       0x2e3, 0x2e6, 0x3, 0x2, 0x2, 0x2, 0x2e4, 0x2e2, 0x3, 0x2, 0x2, 0x2, 
       0x2e4, 0x2e5, 0x3, 0x2, 0x2, 0x2, 0x2e5, 0x91, 0x3, 0x2, 0x2, 0x2, 
       0x2e6, 0x2e4, 0x3, 0x2, 0x2, 0x2, 0x2e7, 0x2ed, 0x5, 0x76, 0x3c, 
       0x2, 0x2e8, 0x2e9, 0x5, 0x76, 0x3c, 0x2, 0x2e9, 0x2ea, 0x5, 0x70, 
       0x39, 0x2, 0x2ea, 0x2eb, 0x5, 0x76, 0x3c, 0x2, 0x2eb, 0x2ed, 0x3, 
       0x2, 0x2, 0x2, 0x2ec, 0x2e7, 0x3, 0x2, 0x2, 0x2, 0x2ec, 0x2e8, 0x3, 
       0x2, 0x2, 0x2, 0x2ed, 0x93, 0x3, 0x2, 0x2, 0x2, 0x2ee, 0x2ef, 0x7, 
       0x6e, 0x2, 0x2, 0x2ef, 0x2f0, 0x5, 0x76, 0x3c, 0x2, 0x2f0, 0x95, 
       0x3, 0x2, 0x2, 0x2, 0x2f1, 0x2f2, 0x9, 0xf, 0x2, 0x2, 0x2f2, 0x97, 
       0x3, 0x2, 0x2, 0x2, 0x2f3, 0x2f4, 0x7, 0x76, 0x2, 0x2, 0x2f4, 0x2f5, 
       0x5, 0x94, 0x4b, 0x2, 0x2f5, 0x2f6, 0x7, 0x6d, 0x2, 0x2, 0x2f6, 0x2f8, 
       0x3, 0x2, 0x2, 0x2, 0x2f7, 0x2f3, 0x3, 0x2, 0x2, 0x2, 0x2f8, 0x2fb, 
       0x3, 0x2, 0x2, 0x2, 0x2f9, 0x2f7, 0x3, 0x2, 0x2, 0x2, 0x2f9, 0x2fa, 
       0x3, 0x2, 0x2, 0x2, 0x2fa, 0x2fc, 0x3, 0x2, 0x2, 0x2, 0x2fb, 0x2f9, 
       0x3, 0x2, 0x2, 0x2, 0x2fc, 0x2fd, 0x7, 0x76, 0x2, 0x2, 0x2fd, 0x2fe, 
       0x5, 0x94, 0x4b, 0x2, 0x2fe, 0x99, 0x3, 0x2, 0x2, 0x2, 0x2ff, 0x300, 
       0x7, 0x76, 0x2, 0x2, 0x300, 0x301, 0x7, 0x4d, 0x2, 0x2, 0x301, 0x302, 
       0x5, 0x9c, 0x4f, 0x2, 0x302, 0x9b, 0x3, 0x2, 0x2, 0x2, 0x303, 0x304, 
       0x7, 0x66, 0x2, 0x2, 0x304, 0x305, 0x5, 0x8e, 0x48, 0x2, 0x305, 0x306, 
       0x7, 0x67, 0x2, 0x2, 0x306, 0x30a, 0x3, 0x2, 0x2, 0x2, 0x307, 0x30a, 
       0x5, 0x50, 0x29, 0x2, 0x308, 0x30a, 0x7, 0x76, 0x2, 0x2, 0x309, 0x303, 
       0x3, 0x2, 0x2, 0x2, 0x309, 0x307, 0x3, 0x2, 0x2, 0x2, 0x309, 0x308, 
       0x3, 0x2, 0x2, 0x2, 0x30a, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x30b, 0x315, 
       0x5, 0xc, 0x7, 0x2, 0x30c, 0x310, 0x7, 0x66, 0x2, 0x2, 0x30d, 0x30f, 
       0x5, 0xc, 0x7, 0x2, 0x30e, 0x30d, 0x3, 0x2, 0x2, 0x2, 0x30f, 0x312, 
       0x3, 0x2, 0x2, 0x2, 0x310, 0x30e, 0x3, 0x2, 0x2, 0x2, 0x310, 0x311, 
       0x3, 0x2, 0x2, 0x2, 0x311, 0x313, 0x3, 0x2, 0x2, 0x2, 0x312, 0x310, 
       0x3, 0x2, 0x2, 0x2, 0x313, 0x315, 0x7, 0x67, 0x2, 0x2, 0x314, 0x30b, 
       0x3, 0x2, 0x2, 0x2, 0x314, 0x30c, 0x3, 0x2, 0x2, 0x2, 0x315, 0x9f, 
       0x3, 0x2, 0x2, 0x2, 0x316, 0x317, 0x7, 0x4e, 0x2, 0x2, 0x317, 0x318, 
       0x7, 0x68, 0x2, 0x2, 0x318, 0x319, 0x5, 0x90, 0x49, 0x2, 0x319, 0x31a, 
       0x7, 0x69, 0x2, 0x2, 0x31a, 0x31d, 0x5, 0x9e, 0x50, 0x2, 0x31b, 0x31c, 
       0x7, 0x4f, 0x2, 0x2, 0x31c, 0x31e, 0x5, 0x9e, 0x50, 0x2, 0x31d, 0x31b, 
       0x3, 0x2, 0x2, 0x2, 0x31d, 0x31e, 0x3, 0x2, 0x2, 0x2, 0x31e, 0xa1, 
       0x3, 0x2, 0x2, 0x2, 0x31f, 0x320, 0x7, 0x50, 0x2, 0x2, 0x320, 0x327, 
       0x5, 0x9a, 0x4e, 0x2, 0x321, 0x322, 0x7, 0x51, 0x2, 0x2, 0x322, 0x323, 
       0x7, 0x68, 0x2, 0x2, 0x323, 0x324, 0x5, 0x90, 0x49, 0x2, 0x324, 0x325, 
       0x7, 0x69, 0x2, 0x2, 0x325, 0x327, 0x3, 0x2, 0x2, 0x2, 0x326, 0x31f, 
       0x3, 0x2, 0x2, 0x2, 0x326, 0x321, 0x3, 0x2, 0x2, 0x2, 0x327, 0xa3, 
       0x3, 0x2, 0x2, 0x2, 0x328, 0x329, 0x5, 0xa2, 0x52, 0x2, 0x329, 0x32a, 
       0x5, 0x9e, 0x50, 0x2, 0x32a, 0xa5, 0x3, 0x2, 0x2, 0x2, 0x32b, 0x32c, 
       0x7, 0x50, 0x2, 0x2, 0x32c, 0x32d, 0x7, 0x68, 0x2, 0x2, 0x32d, 0x32e, 
       0x5, 0x34, 0x1b, 0x2, 0x32e, 0x32f, 0x7, 0x76, 0x2, 0x2, 0x32f, 0x330, 
       0x5, 0x94, 0x4b, 0x2, 0x330, 0x331, 0x7, 0x6b, 0x2, 0x2, 0x331, 0x332, 
       0x5, 0x90, 0x49, 0x2, 0x332, 0x333, 0x7, 0x6b, 0x2, 0x2, 0x333, 0x334, 
       0x5, 0x76, 0x3c, 0x2, 0x334, 0x335, 0x7, 0x69, 0x2, 0x2, 0x335, 0x336, 
       0x7, 0x66, 0x2, 0x2, 0x336, 0x337, 0x5, 0x9e, 0x50, 0x2, 0x337, 0x338, 
       0x7, 0x67, 0x2, 0x2, 0x338, 0x348, 0x3, 0x2, 0x2, 0x2, 0x339, 0x33a, 
       0x7, 0x50, 0x2, 0x2, 0x33a, 0x33b, 0x7, 0x68, 0x2, 0x2, 0x33b, 0x33c, 
       0x5, 0x34, 0x1b, 0x2, 0x33c, 0x33d, 0x7, 0x76, 0x2, 0x2, 0x33d, 0x33e, 
       0x7, 0x6a, 0x2, 0x2, 0x33e, 0x33f, 0x7, 0x52, 0x2, 0x2, 0x33f, 0x340, 
       0x7, 0x68, 0x2, 0x2, 0x340, 0x341, 0x5, 0x76, 0x3c, 0x2, 0x341, 0x342, 
       0x7, 0x69, 0x2, 0x2, 0x342, 0x343, 0x7, 0x69, 0x2, 0x2, 0x343, 0x344, 
       0x7, 0x66, 0x2, 0x2, 0x344, 0x345, 0x5, 0x9e, 0x50, 0x2, 0x345, 0x346, 
       0x7, 0x67, 0x2, 0x2, 0x346, 0x348, 0x3, 0x2, 0x2, 0x2, 0x347, 0x32b, 
       0x3, 0x2, 0x2, 0x2, 0x347, 0x339, 0x3, 0x2, 0x2, 0x2, 0x348, 0xa7, 
       0x3, 0x2, 0x2, 0x2, 0x349, 0x34a, 0x5, 0xaa, 0x56, 0x2, 0x34a, 0x34b, 
       0x7, 0x6b, 0x2, 0x2, 0x34b, 0xa9, 0x3, 0x2, 0x2, 0x2, 0x34c, 0x34d, 
       0x9, 0x10, 0x2, 0x2, 0x34d, 0xab, 0x3, 0x2, 0x2, 0x2, 0x34e, 0x34f, 
       0x7, 0x56, 0x2, 0x2, 0x34f, 0x355, 0x7, 0x76, 0x2, 0x2, 0x350, 0x352, 
       0x7, 0x68, 0x2, 0x2, 0x351, 0x353, 0x5, 0x42, 0x22, 0x2, 0x352, 0x351, 
       0x3, 0x2, 0x2, 0x2, 0x352, 0x353, 0x3, 0x2, 0x2, 0x2, 0x353, 0x354, 
       0x3, 0x2, 0x2, 0x2, 0x354, 0x356, 0x7, 0x69, 0x2, 0x2, 0x355, 0x350, 
       0x3, 0x2, 0x2, 0x2, 0x355, 0x356, 0x3, 0x2, 0x2, 0x2, 0x356, 0x358, 
       0x3, 0x2, 0x2, 0x2, 0x357, 0x359, 0x5, 0x1a, 0xe, 0x2, 0x358, 0x357, 
       0x3, 0x2, 0x2, 0x2, 0x358, 0x359, 0x3, 0x2, 0x2, 0x2, 0x359, 0x35b, 
       0x3, 0x2, 0x2, 0x2, 0x35a, 0x35c, 0x5, 0x34, 0x1b, 0x2, 0x35b, 0x35a, 
       0x3, 0x2, 0x2, 0x2, 0x35b, 0x35c, 0x3, 0x2, 0x2, 0x2, 0x35c, 0x35d, 
       0x3, 0x2, 0x2, 0x2, 0x35d, 0x35e, 0x7, 0x6b, 0x2, 0x2, 0x35e, 0xad, 
       0x3, 0x2, 0x2, 0x2, 0x35f, 0x360, 0x7, 0x76, 0x2, 0x2, 0x360, 0x362, 
       0x7, 0x68, 0x2, 0x2, 0x361, 0x363, 0x5, 0x8e, 0x48, 0x2, 0x362, 0x361, 
       0x3, 0x2, 0x2, 0x2, 0x362, 0x363, 0x3, 0x2, 0x2, 0x2, 0x363, 0x364, 
       0x3, 0x2, 0x2, 0x2, 0x364, 0x365, 0x7, 0x69, 0x2, 0x2, 0x365, 0xaf, 
       0x3, 0x2, 0x2, 0x2, 0x366, 0x367, 0x7, 0x57, 0x2, 0x2, 0x367, 0x36d, 
       0x7, 0x76, 0x2, 0x2, 0x368, 0x36a, 0x7, 0x68, 0x2, 0x2, 0x369, 0x36b, 
       0x5, 0x46, 0x24, 0x2, 0x36a, 0x369, 0x3, 0x2, 0x2, 0x2, 0x36a, 0x36b, 
       0x3, 0x2, 0x2, 0x2, 0x36b, 0x36c, 0x3, 0x2, 0x2, 0x2, 0x36c, 0x36e, 
       0x7, 0x69, 0x2, 0x2, 0x36d, 0x368, 0x3, 0x2, 0x2, 0x2, 0x36d, 0x36e, 
       0x3, 0x2, 0x2, 0x2, 0x36e, 0x370, 0x3, 0x2, 0x2, 0x2, 0x36f, 0x371, 
       0x5, 0x2a, 0x16, 0x2, 0x370, 0x36f, 0x3, 0x2, 0x2, 0x2, 0x370, 0x371, 
       0x3, 0x2, 0x2, 0x2, 0x371, 0x373, 0x3, 0x2, 0x2, 0x2, 0x372, 0x374, 
       0x5, 0x1a, 0xe, 0x2, 0x373, 0x372, 0x3, 0x2, 0x2, 0x2, 0x373, 0x374, 
       0x3, 0x2, 0x2, 0x2, 0x374, 0x375, 0x3, 0x2, 0x2, 0x2, 0x375, 0x376, 
       0x5, 0xb4, 0x5b, 0x2, 0x376, 0xb1, 0x3, 0x2, 0x2, 0x2, 0x377, 0x378, 
       0x7, 0x58, 0x2, 0x2, 0x378, 0x379, 0x5, 0xc, 0x7, 0x2, 0x379, 0xb3, 
       0x3, 0x2, 0x2, 0x2, 0x37a, 0x37e, 0x7, 0x66, 0x2, 0x2, 0x37b, 0x37d, 
       0x5, 0xc, 0x7, 0x2, 0x37c, 0x37b, 0x3, 0x2, 0x2, 0x2, 0x37d, 0x380, 
       0x3, 0x2, 0x2, 0x2, 0x37e, 0x37c, 0x3, 0x2, 0x2, 0x2, 0x37e, 0x37f, 
       0x3, 0x2, 0x2, 0x2, 0x37f, 0x382, 0x3, 0x2, 0x2, 0x2, 0x380, 0x37e, 
       0x3, 0x2, 0x2, 0x2, 0x381, 0x383, 0x5, 0xb2, 0x5a, 0x2, 0x382, 0x381, 
       0x3, 0x2, 0x2, 0x2, 0x382, 0x383, 0x3, 0x2, 0x2, 0x2, 0x383, 0x384, 
       0x3, 0x2, 0x2, 0x2, 0x384, 0x388, 0x7, 0x67, 0x2, 0x2, 0x385, 0x386, 
       0x7, 0x71, 0x2, 0x2, 0x386, 0x388, 0x7, 0x6b, 0x2, 0x2, 0x387, 0x37a, 
       0x3, 0x2, 0x2, 0x2, 0x387, 0x385, 0x3, 0x2, 0x2, 0x2, 0x388, 0xb5, 
       0x3, 0x2, 0x2, 0x2, 0x389, 0x38a, 0x7, 0x76, 0x2, 0x2, 0x38a, 0x38c, 
       0x7, 0x68, 0x2, 0x2, 0x38b, 0x38d, 0x5, 0x8e, 0x48, 0x2, 0x38c, 0x38b, 
       0x3, 0x2, 0x2, 0x2, 0x38c, 0x38d, 0x3, 0x2, 0x2, 0x2, 0x38d, 0x38e, 
       0x3, 0x2, 0x2, 0x2, 0x38e, 0x38f, 0x7, 0x69, 0x2, 0x2, 0x38f, 0x390, 
       0x3, 0x2, 0x2, 0x2, 0x390, 0x391, 0x5, 0x8e, 0x48, 0x2, 0x391, 0xb7, 
       0x3, 0x2, 0x2, 0x2, 0x392, 0x393, 0x7, 0x59, 0x2, 0x2, 0x393, 0x397, 
       0x7, 0x66, 0x2, 0x2, 0x394, 0x396, 0x5, 0xc, 0x7, 0x2, 0x395, 0x394, 
       0x3, 0x2, 0x2, 0x2, 0x396, 0x399, 0x3, 0x2, 0x2, 0x2, 0x397, 0x395, 
       0x3, 0x2, 0x2, 0x2, 0x397, 0x398, 0x3, 0x2, 0x2, 0x2, 0x398, 0x39a, 
       0x3, 0x2, 0x2, 0x2, 0x399, 0x397, 0x3, 0x2, 0x2, 0x2, 0x39a, 0x39b, 
       0x7, 0x67, 0x2, 0x2, 0x39b, 0xb9, 0x3, 0x2, 0x2, 0x2, 0x39c, 0x3a2, 
       0x7, 0x5a, 0x2, 0x2, 0x39d, 0x39f, 0x7, 0x5b, 0x2, 0x2, 0x39e, 0x3a0, 
       0x7, 0x75, 0x2, 0x2, 0x39f, 0x39e, 0x3, 0x2, 0x2, 0x2, 0x39f, 0x3a0, 
       0x3, 0x2, 0x2, 0x2, 0x3a0, 0x3a2, 0x3, 0x2, 0x2, 0x2, 0x3a1, 0x39c, 
       0x3, 0x2, 0x2, 0x2, 0x3a1, 0x39d, 0x3, 0x2, 0x2, 0x2, 0x3a2, 0xbb, 
       0x3, 0x2, 0x2, 0x2, 0x3a3, 0x3a4, 0x7, 0x5c, 0x2, 0x2, 0x3a4, 0x3a5, 
       0x7, 0x76, 0x2, 0x2, 0x3a5, 0x3aa, 0x5, 0x56, 0x2c, 0x2, 0x3a6, 0x3a7, 
       0x7, 0x5d, 0x2, 0x2, 0x3a7, 0x3a8, 0x7, 0x78, 0x2, 0x2, 0x3a8, 0x3aa, 
       0x5, 0x56, 0x2c, 0x2, 0x3a9, 0x3a3, 0x3, 0x2, 0x2, 0x2, 0x3a9, 0x3a6, 
       0x3, 0x2, 0x2, 0x2, 0x3aa, 0xbd, 0x3, 0x2, 0x2, 0x2, 0x3ab, 0x3ae, 
       0x5, 0xc0, 0x61, 0x2, 0x3ac, 0x3ae, 0x7, 0x5e, 0x2, 0x2, 0x3ad, 0x3ab, 
       0x3, 0x2, 0x2, 0x2, 0x3ad, 0x3ac, 0x3, 0x2, 0x2, 0x2, 0x3ae, 0xbf, 
       0x3, 0x2, 0x2, 0x2, 0x3af, 0x3b8, 0x7, 0x78, 0x2, 0x2, 0x3b0, 0x3b1, 
       0x7, 0x42, 0x2, 0x2, 0x3b1, 0x3b4, 0x7, 0x68, 0x2, 0x2, 0x3b2, 0x3b5, 
       0x7, 0x76, 0x2, 0x2, 0x3b3, 0x3b5, 0x5, 0x56, 0x2c, 0x2, 0x3b4, 0x3b2, 
       0x3, 0x2, 0x2, 0x2, 0x3b4, 0x3b3, 0x3, 0x2, 0x2, 0x2, 0x3b5, 0x3b6, 
       0x3, 0x2, 0x2, 0x2, 0x3b6, 0x3b8, 0x7, 0x69, 0x2, 0x2, 0x3b7, 0x3af, 
       0x3, 0x2, 0x2, 0x2, 0x3b7, 0x3b0, 0x3, 0x2, 0x2, 0x2, 0x3b8, 0xc1, 
       0x3, 0x2, 0x2, 0x2, 0x3b9, 0x3ba, 0x9, 0x11, 0x2, 0x2, 0x3ba, 0xc3, 
       0x3, 0x2, 0x2, 0x2, 0x3bb, 0x3c1, 0x5, 0xc2, 0x62, 0x2, 0x3bc, 0x3be, 
       0x7, 0x68, 0x2, 0x2, 0x3bd, 0x3bf, 0x5, 0x8e, 0x48, 0x2, 0x3be, 0x3bd, 
       0x3, 0x2, 0x2, 0x2, 0x3be, 0x3bf, 0x3, 0x2, 0x2, 0x2, 0x3bf, 0x3c0, 
       0x3, 0x2, 0x2, 0x2, 0x3c0, 0x3c2, 0x7, 0x69, 0x2, 0x2, 0x3c1, 0x3bc, 
       0x3, 0x2, 0x2, 0x2, 0x3c1, 0x3c2, 0x3, 0x2, 0x2, 0x2, 0x3c2, 0x3c3, 
       0x3, 0x2, 0x2, 0x2, 0x3c3, 0x3c4, 0x5, 0x1c, 0xf, 0x2, 0x3c4, 0x3c5, 
       0x5, 0x4c, 0x27, 0x2, 0x3c5, 0xc5, 0x3, 0x2, 0x2, 0x2, 0x3c6, 0x3c7, 
       0x5, 0xc4, 0x63, 0x2, 0x3c7, 0x3c8, 0x7, 0x6b, 0x2, 0x2, 0x3c8, 0x3cb, 
       0x3, 0x2, 0x2, 0x2, 0x3c9, 0x3cb, 0x5, 0xbc, 0x5f, 0x2, 0x3ca, 0x3c6, 
       0x3, 0x2, 0x2, 0x2, 0x3ca, 0x3c9, 0x3, 0x2, 0x2, 0x2, 0x3cb, 0xc7, 
       0x3, 0x2, 0x2, 0x2, 0x3cc, 0x3cf, 0x5, 0xca, 0x66, 0x2, 0x3cd, 0x3cf, 
       0x5, 0xcc, 0x67, 0x2, 0x3ce, 0x3cc, 0x3, 0x2, 0x2, 0x2, 0x3ce, 0x3cd, 
       0x3, 0x2, 0x2, 0x2, 0x3cf, 0xc9, 0x3, 0x2, 0x2, 0x2, 0x3d0, 0x3d1, 
       0x7, 0x61, 0x2, 0x2, 0x3d1, 0x3d2, 0x5, 0xce, 0x68, 0x2, 0x3d2, 0x3d3, 
       0x7, 0x6b, 0x2, 0x2, 0x3d3, 0xcb, 0x3, 0x2, 0x2, 0x2, 0x3d4, 0x3d5, 
       0x7, 0x62, 0x2, 0x2, 0x3d5, 0x3db, 0x7, 0x76, 0x2, 0x2, 0x3d6, 0x3d8, 
       0x7, 0x68, 0x2, 0x2, 0x3d7, 0x3d9, 0x5, 0xd0, 0x69, 0x2, 0x3d8, 0x3d7, 
       0x3, 0x2, 0x2, 0x2, 0x3d8, 0x3d9, 0x3, 0x2, 0x2, 0x2, 0x3d9, 0x3da, 
       0x3, 0x2, 0x2, 0x2, 0x3da, 0x3dc, 0x7, 0x69, 0x2, 0x2, 0x3db, 0x3d6, 
       0x3, 0x2, 0x2, 0x2, 0x3db, 0x3dc, 0x3, 0x2, 0x2, 0x2, 0x3dc, 0x3dd, 
       0x3, 0x2, 0x2, 0x2, 0x3dd, 0x3df, 0x5, 0x20, 0x11, 0x2, 0x3de, 0x3e0, 
       0x5, 0x1a, 0xe, 0x2, 0x3df, 0x3de, 0x3, 0x2, 0x2, 0x2, 0x3df, 0x3e0, 
       0x3, 0x2, 0x2, 0x2, 0x3e0, 0x3e1, 0x3, 0x2, 0x2, 0x2, 0x3e1, 0x3e5, 
       0x7, 0x66, 0x2, 0x2, 0x3e2, 0x3e4, 0xb, 0x2, 0x2, 0x2, 0x3e3, 0x3e2, 
       0x3, 0x2, 0x2, 0x2, 0x3e4, 0x3e7, 0x3, 0x2, 0x2, 0x2, 0x3e5, 0x3e6, 
       0x3, 0x2, 0x2, 0x2, 0x3e5, 0x3e3, 0x3, 0x2, 0x2, 0x2, 0x3e6, 0x3e8, 
       0x3, 0x2, 0x2, 0x2, 0x3e7, 0x3e5, 0x3, 0x2, 0x2, 0x2, 0x3e8, 0x3e9, 
       0x7, 0x67, 0x2, 0x2, 0x3e9, 0xcd, 0x3, 0x2, 0x2, 0x2, 0x3ea, 0x3eb, 
       0x9, 0x12, 0x2, 0x2, 0x3eb, 0xcf, 0x3, 0x2, 0x2, 0x2, 0x3ec, 0x3ef, 
       0x5, 0x46, 0x24, 0x2, 0x3ed, 0x3ef, 0x5, 0x8e, 0x48, 0x2, 0x3ee, 
       0x3ec, 0x3, 0x2, 0x2, 0x2, 0x3ee, 0x3ed, 0x3, 0x2, 0x2, 0x2, 0x3ef, 
       0xd1, 0x3, 0x2, 0x2, 0x2, 0x66, 0xd5, 0xd7, 0xdb, 0xe0, 0xf1, 0xfe, 
       0x110, 0x118, 0x11c, 0x131, 0x13e, 0x142, 0x145, 0x149, 0x152, 0x163, 
       0x16e, 0x170, 0x179, 0x17f, 0x184, 0x189, 0x18f, 0x196, 0x1a3, 0x1b6, 
       0x1b8, 0x1bf, 0x1c7, 0x1d2, 0x1da, 0x1de, 0x1e2, 0x1ed, 0x1f0, 0x1f8, 
       0x1fa, 0x207, 0x20b, 0x211, 0x217, 0x224, 0x22a, 0x236, 0x23d, 0x240, 
       0x24b, 0x25a, 0x261, 0x26c, 0x277, 0x282, 0x28d, 0x293, 0x299, 0x29d, 
       0x2b3, 0x2bc, 0x2be, 0x2c5, 0x2d4, 0x2dc, 0x2e4, 0x2ec, 0x2f9, 0x309, 
       0x310, 0x314, 0x31d, 0x326, 0x347, 0x352, 0x355, 0x358, 0x35b, 0x362, 
       0x36a, 0x36d, 0x370, 0x373, 0x37e, 0x382, 0x387, 0x38c, 0x397, 0x39f, 
       0x3a1, 0x3a9, 0x3ad, 0x3b4, 0x3b7, 0x3be, 0x3c1, 0x3ca, 0x3ce, 0x3d8, 
       0x3db, 0x3df, 0x3e5, 0x3ee, 
  };

  _serializedATN.insert(_serializedATN.end(), serializedATNSegment0,
    serializedATNSegment0 + sizeof(serializedATNSegment0) / sizeof(serializedATNSegment0[0]));


  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

qasm3Parser::Initializer qasm3Parser::_init;

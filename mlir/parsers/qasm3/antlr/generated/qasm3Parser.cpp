
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
    setState(206);
    header();
    setState(211);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__2)
      | (1ULL << qasm3Parser::T__3)
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
      | (1ULL << qasm3Parser::T__18)
      | (1ULL << qasm3Parser::T__19)
      | (1ULL << qasm3Parser::T__20)
      | (1ULL << qasm3Parser::T__21)
      | (1ULL << qasm3Parser::T__22)
      | (1ULL << qasm3Parser::T__23)
      | (1ULL << qasm3Parser::T__24)
      | (1ULL << qasm3Parser::T__25)
      | (1ULL << qasm3Parser::T__26)
      | (1ULL << qasm3Parser::T__28)
      | (1ULL << qasm3Parser::T__29)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__49)
      | (1ULL << qasm3Parser::T__50)
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
      | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 74) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 74)) & ((1ULL << (qasm3Parser::T__73 - 74))
      | (1ULL << (qasm3Parser::T__75 - 74))
      | (1ULL << (qasm3Parser::T__76 - 74))
      | (1ULL << (qasm3Parser::T__78 - 74))
      | (1ULL << (qasm3Parser::T__79 - 74))
      | (1ULL << (qasm3Parser::T__80 - 74))
      | (1ULL << (qasm3Parser::T__81 - 74))
      | (1ULL << (qasm3Parser::T__82 - 74))
      | (1ULL << (qasm3Parser::T__83 - 74))
      | (1ULL << (qasm3Parser::T__84 - 74))
      | (1ULL << (qasm3Parser::T__85 - 74))
      | (1ULL << (qasm3Parser::T__86 - 74))
      | (1ULL << (qasm3Parser::T__87 - 74))
      | (1ULL << (qasm3Parser::T__88 - 74))
      | (1ULL << (qasm3Parser::T__89 - 74))
      | (1ULL << (qasm3Parser::T__90 - 74))
      | (1ULL << (qasm3Parser::T__91 - 74))
      | (1ULL << (qasm3Parser::T__92 - 74))
      | (1ULL << (qasm3Parser::T__93 - 74))
      | (1ULL << (qasm3Parser::LPAREN - 74))
      | (1ULL << (qasm3Parser::MINUS - 74))
      | (1ULL << (qasm3Parser::Constant - 74))
      | (1ULL << (qasm3Parser::Integer - 74))
      | (1ULL << (qasm3Parser::Identifier - 74))
      | (1ULL << (qasm3Parser::RealNumber - 74))
      | (1ULL << (qasm3Parser::TimingLiteral - 74))
      | (1ULL << (qasm3Parser::StringLiteral - 74)))) != 0)) {
      setState(209);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case qasm3Parser::T__3:
        case qasm3Parser::T__4:
        case qasm3Parser::T__18:
        case qasm3Parser::T__81:
        case qasm3Parser::T__82:
        case qasm3Parser::T__84:
        case qasm3Parser::T__92:
        case qasm3Parser::T__93: {
          setState(207);
          globalStatement();
          break;
        }

        case qasm3Parser::T__2:
        case qasm3Parser::T__5:
        case qasm3Parser::T__6:
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
        case qasm3Parser::T__19:
        case qasm3Parser::T__20:
        case qasm3Parser::T__21:
        case qasm3Parser::T__22:
        case qasm3Parser::T__23:
        case qasm3Parser::T__24:
        case qasm3Parser::T__25:
        case qasm3Parser::T__26:
        case qasm3Parser::T__28:
        case qasm3Parser::T__29:
        case qasm3Parser::T__30:
        case qasm3Parser::T__49:
        case qasm3Parser::T__50:
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
        case qasm3Parser::T__73:
        case qasm3Parser::T__75:
        case qasm3Parser::T__76:
        case qasm3Parser::T__78:
        case qasm3Parser::T__79:
        case qasm3Parser::T__80:
        case qasm3Parser::T__83:
        case qasm3Parser::T__85:
        case qasm3Parser::T__86:
        case qasm3Parser::T__87:
        case qasm3Parser::T__88:
        case qasm3Parser::T__89:
        case qasm3Parser::T__90:
        case qasm3Parser::T__91:
        case qasm3Parser::LPAREN:
        case qasm3Parser::MINUS:
        case qasm3Parser::Constant:
        case qasm3Parser::Integer:
        case qasm3Parser::Identifier:
        case qasm3Parser::RealNumber:
        case qasm3Parser::TimingLiteral:
        case qasm3Parser::StringLiteral: {
          setState(208);
          statement();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(213);
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
    setState(215);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::T__0) {
      setState(214);
      version();
    }
    setState(220);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == qasm3Parser::T__1) {
      setState(217);
      include();
      setState(222);
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
    setState(223);
    match(qasm3Parser::T__0);
    setState(224);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::Integer

    || _la == qasm3Parser::RealNumber)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(225);
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
    setState(227);
    match(qasm3Parser::T__1);
    setState(228);
    match(qasm3Parser::StringLiteral);
    setState(229);
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
    setState(237);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__82: {
        enterOuterAlt(_localctx, 1);
        setState(231);
        subroutineDefinition();
        break;
      }

      case qasm3Parser::T__81: {
        enterOuterAlt(_localctx, 2);
        setState(232);
        kernelDeclaration();
        break;
      }

      case qasm3Parser::T__18: {
        enterOuterAlt(_localctx, 3);
        setState(233);
        quantumGateDefinition();
        break;
      }

      case qasm3Parser::T__92:
      case qasm3Parser::T__93: {
        enterOuterAlt(_localctx, 4);
        setState(234);
        calibration();
        break;
      }

      case qasm3Parser::T__3:
      case qasm3Parser::T__4: {
        enterOuterAlt(_localctx, 5);
        setState(235);
        quantumDeclarationStatement();
        break;
      }

      case qasm3Parser::T__84: {
        enterOuterAlt(_localctx, 6);
        setState(236);
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
    setState(249);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(239);
      expressionStatement();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(240);
      assignmentStatement();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(241);
      classicalDeclarationStatement();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(242);
      branchingStatement();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(243);
      loopStatement();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(244);
      controlDirectiveStatement();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(245);
      aliasStatement();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(246);
      quantumStatement();
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(247);
      returnStatement();
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(248);
      qcor_test_statement();
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
  enterRule(_localctx, 12, qasm3Parser::RuleQcor_test_statement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(251);
    match(qasm3Parser::T__2);
    setState(252);
    match(qasm3Parser::LPAREN);
    setState(253);
    booleanExpression(0);
    setState(254);
    match(qasm3Parser::RPAREN);
    setState(255);
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
  enterRule(_localctx, 14, qasm3Parser::RuleQuantumDeclarationStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(257);
    quantumDeclaration();
    setState(258);
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
  enterRule(_localctx, 16, qasm3Parser::RuleClassicalDeclarationStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(262);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__5:
      case qasm3Parser::T__6:
      case qasm3Parser::T__7:
      case qasm3Parser::T__8:
      case qasm3Parser::T__9:
      case qasm3Parser::T__10:
      case qasm3Parser::T__11:
      case qasm3Parser::T__12:
      case qasm3Parser::T__13:
      case qasm3Parser::T__14:
      case qasm3Parser::T__85:
      case qasm3Parser::T__86: {
        setState(260);
        classicalDeclaration();
        break;
      }

      case qasm3Parser::T__15: {
        setState(261);
        constantDeclaration();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(264);
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
  enterRule(_localctx, 18, qasm3Parser::RuleClassicalAssignment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(266);
    indexIdentifier(0);
    setState(267);
    assignmentOperator();
    setState(270);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      setState(268);
      expression(0);
      break;
    }

    case 2: {
      setState(269);
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
  enterRule(_localctx, 20, qasm3Parser::RuleAssignmentStatement);

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
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      setState(272);
      classicalAssignment();
      break;
    }

    case 2: {
      setState(273);
      quantumMeasurementAssignment();
      break;
    }

    default:
      break;
    }
    setState(276);
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
  enterRule(_localctx, 22, qasm3Parser::RuleReturnSignature);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(278);
    match(qasm3Parser::ARROW);
    setState(279);
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
  enterRule(_localctx, 24, qasm3Parser::RuleDesignator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(281);
    match(qasm3Parser::LBRACKET);
    setState(282);
    expression(0);
    setState(283);
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
  enterRule(_localctx, 26, qasm3Parser::RuleDoubleDesignator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(285);
    match(qasm3Parser::LBRACKET);
    setState(286);
    expression(0);
    setState(287);
    match(qasm3Parser::COMMA);
    setState(288);
    expression(0);
    setState(289);
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
  enterRule(_localctx, 28, qasm3Parser::RuleIdentifierList);

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
    setState(295);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(291);
        match(qasm3Parser::Identifier);
        setState(292);
        match(qasm3Parser::COMMA); 
      }
      setState(297);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    }
    setState(298);
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
  enterRule(_localctx, 30, qasm3Parser::RuleAssociation);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(300);
    match(qasm3Parser::COLON);
    setState(301);
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
  enterRule(_localctx, 32, qasm3Parser::RuleQuantumType);
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
    setState(303);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__3

    || _la == qasm3Parser::T__4)) {
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

qasm3Parser::QuantumTypeContext* qasm3Parser::QuantumDeclarationContext::quantumType() {
  return getRuleContext<qasm3Parser::QuantumTypeContext>(0);
}

qasm3Parser::IndexIdentifierListContext* qasm3Parser::QuantumDeclarationContext::indexIdentifierList() {
  return getRuleContext<qasm3Parser::IndexIdentifierListContext>(0);
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
  enterRule(_localctx, 34, qasm3Parser::RuleQuantumDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(305);
    quantumType();
    setState(306);
    indexIdentifierList();
   
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
  enterRule(_localctx, 36, qasm3Parser::RuleQuantumArgument);
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
    setState(308);
    quantumType();
    setState(310);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LBRACKET) {
      setState(309);
      designator();
    }
    setState(312);
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
  enterRule(_localctx, 38, qasm3Parser::RuleQuantumArgumentList);

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
    setState(319);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(314);
        quantumArgument();
        setState(315);
        match(qasm3Parser::COMMA); 
      }
      setState(321);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
    }
    setState(322);
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
  enterRule(_localctx, 40, qasm3Parser::RuleBitType);
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
    setState(324);
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
  enterRule(_localctx, 42, qasm3Parser::RuleSingleDesignatorType);
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
    setState(326);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10))) != 0))) {
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
  enterRule(_localctx, 44, qasm3Parser::RuleDoubleDesignatorType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(328);
    match(qasm3Parser::T__11);
   
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
  enterRule(_localctx, 46, qasm3Parser::RuleNoDesignatorType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(336);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__12: {
        enterOuterAlt(_localctx, 1);
        setState(330);
        match(qasm3Parser::T__12);
        break;
      }

      case qasm3Parser::T__85:
      case qasm3Parser::T__86: {
        enterOuterAlt(_localctx, 2);
        setState(331);
        timingType();
        break;
      }

      case qasm3Parser::T__7: {
        enterOuterAlt(_localctx, 3);
        setState(332);
        match(qasm3Parser::T__7);
        break;
      }

      case qasm3Parser::T__13: {
        enterOuterAlt(_localctx, 4);
        setState(333);
        match(qasm3Parser::T__13);
        break;
      }

      case qasm3Parser::T__9: {
        enterOuterAlt(_localctx, 5);
        setState(334);
        match(qasm3Parser::T__9);
        break;
      }

      case qasm3Parser::T__14: {
        enterOuterAlt(_localctx, 6);
        setState(335);
        match(qasm3Parser::T__14);
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
  enterRule(_localctx, 48, qasm3Parser::RuleClassicalType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(349);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(338);
      singleDesignatorType();
      setState(339);
      designator();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(341);
      doubleDesignatorType();
      setState(342);
      doubleDesignator();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(344);
      noDesignatorType();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(345);
      bitType();
      setState(347);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == qasm3Parser::LBRACKET) {
        setState(346);
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
  enterRule(_localctx, 50, qasm3Parser::RuleConstantDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(351);
    match(qasm3Parser::T__15);
    setState(352);
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
  enterRule(_localctx, 52, qasm3Parser::RuleSingleDesignatorDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(354);
    singleDesignatorType();
    setState(355);
    designator();
    setState(358);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx)) {
    case 1: {
      setState(356);
      identifierList();
      break;
    }

    case 2: {
      setState(357);
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
  enterRule(_localctx, 54, qasm3Parser::RuleDoubleDesignatorDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(360);
    doubleDesignatorType();
    setState(361);
    doubleDesignator();
    setState(364);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
    case 1: {
      setState(362);
      identifierList();
      break;
    }

    case 2: {
      setState(363);
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
  enterRule(_localctx, 56, qasm3Parser::RuleNoDesignatorDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(366);
    noDesignatorType();
    setState(369);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx)) {
    case 1: {
      setState(367);
      identifierList();
      break;
    }

    case 2: {
      setState(368);
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
  enterRule(_localctx, 58, qasm3Parser::RuleBitDeclaration);

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
    bitType();
    setState(374);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
    case 1: {
      setState(372);
      indexIdentifierList();
      break;
    }

    case 2: {
      setState(373);
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
  enterRule(_localctx, 60, qasm3Parser::RuleClassicalDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(380);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(376);
      singleDesignatorDeclaration();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(377);
      doubleDesignatorDeclaration();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(378);
      noDesignatorDeclaration();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(379);
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
  enterRule(_localctx, 62, qasm3Parser::RuleClassicalTypeList);

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
    setState(387);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(382);
        classicalType();
        setState(383);
        match(qasm3Parser::COMMA); 
      }
      setState(389);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx);
    }
    setState(390);
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
  enterRule(_localctx, 64, qasm3Parser::RuleClassicalArgument);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(392);
    classicalType();
    setState(393);
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
  enterRule(_localctx, 66, qasm3Parser::RuleClassicalArgumentList);

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
    setState(400);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(395);
        classicalArgument();
        setState(396);
        match(qasm3Parser::COMMA); 
      }
      setState(402);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx);
    }
    setState(403);
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
  enterRule(_localctx, 68, qasm3Parser::RuleAliasStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(405);
    match(qasm3Parser::T__16);
    setState(406);
    match(qasm3Parser::Identifier);
    setState(407);
    match(qasm3Parser::EQUALS);
    setState(408);
    indexIdentifier(0);
    setState(409);
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
  size_t startState = 70;
  enterRecursionRule(_localctx, 70, qasm3Parser::RuleIndexIdentifier, precedence);

    

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
    setState(421);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
    case 1: {
      setState(412);
      match(qasm3Parser::Identifier);
      setState(413);
      rangeDefinition();
      break;
    }

    case 2: {
      setState(414);
      match(qasm3Parser::Identifier);
      setState(419);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 22, _ctx)) {
      case 1: {
        setState(415);
        match(qasm3Parser::LBRACKET);
        setState(416);
        expressionList();
        setState(417);
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
    setState(428);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<IndexIdentifierContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleIndexIdentifier);
        setState(423);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(424);
        match(qasm3Parser::T__17);
        setState(425);
        indexIdentifier(2); 
      }
      setState(430);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
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
  enterRule(_localctx, 72, qasm3Parser::RuleIndexIdentifierList);

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
    setState(436);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(431);
        indexIdentifier(0);
        setState(432);
        match(qasm3Parser::COMMA); 
      }
      setState(438);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
    }
    setState(439);
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
  enterRule(_localctx, 74, qasm3Parser::RuleIndexEqualsAssignmentList);

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
    setState(447);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 26, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(441);
        indexIdentifier(0);
        setState(442);
        equalsExpression();
        setState(443);
        match(qasm3Parser::COMMA); 
      }
      setState(449);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 26, _ctx);
    }
    setState(450);
    indexIdentifier(0);
    setState(451);
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
  enterRule(_localctx, 76, qasm3Parser::RuleRangeDefinition);
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
    setState(453);
    match(qasm3Parser::LBRACKET);
    setState(455);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
      | (1ULL << qasm3Parser::T__6)
      | (1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__29)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__49)
      | (1ULL << qasm3Parser::T__50)
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
      | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 86) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 86)) & ((1ULL << (qasm3Parser::T__85 - 86))
      | (1ULL << (qasm3Parser::T__86 - 86))
      | (1ULL << (qasm3Parser::T__89 - 86))
      | (1ULL << (qasm3Parser::LPAREN - 86))
      | (1ULL << (qasm3Parser::MINUS - 86))
      | (1ULL << (qasm3Parser::Constant - 86))
      | (1ULL << (qasm3Parser::Integer - 86))
      | (1ULL << (qasm3Parser::Identifier - 86))
      | (1ULL << (qasm3Parser::RealNumber - 86))
      | (1ULL << (qasm3Parser::TimingLiteral - 86))
      | (1ULL << (qasm3Parser::StringLiteral - 86)))) != 0)) {
      setState(454);
      expression(0);
    }
    setState(457);
    match(qasm3Parser::COLON);
    setState(459);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
      | (1ULL << qasm3Parser::T__6)
      | (1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__29)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__49)
      | (1ULL << qasm3Parser::T__50)
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
      | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 86) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 86)) & ((1ULL << (qasm3Parser::T__85 - 86))
      | (1ULL << (qasm3Parser::T__86 - 86))
      | (1ULL << (qasm3Parser::T__89 - 86))
      | (1ULL << (qasm3Parser::LPAREN - 86))
      | (1ULL << (qasm3Parser::MINUS - 86))
      | (1ULL << (qasm3Parser::Constant - 86))
      | (1ULL << (qasm3Parser::Integer - 86))
      | (1ULL << (qasm3Parser::Identifier - 86))
      | (1ULL << (qasm3Parser::RealNumber - 86))
      | (1ULL << (qasm3Parser::TimingLiteral - 86))
      | (1ULL << (qasm3Parser::StringLiteral - 86)))) != 0)) {
      setState(458);
      expression(0);
    }
    setState(463);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::COLON) {
      setState(461);
      match(qasm3Parser::COLON);
      setState(462);
      expression(0);
    }
    setState(465);
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
  enterRule(_localctx, 78, qasm3Parser::RuleQuantumGateDefinition);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(467);
    match(qasm3Parser::T__18);
    setState(468);
    quantumGateSignature();
    setState(469);
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
  enterRule(_localctx, 80, qasm3Parser::RuleQuantumGateSignature);
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
    setState(471);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__19

    || _la == qasm3Parser::T__20 || _la == qasm3Parser::Identifier)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(477);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(472);
      match(qasm3Parser::LPAREN);
      setState(474);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == qasm3Parser::Identifier) {
        setState(473);
        identifierList();
      }
      setState(476);
      match(qasm3Parser::RPAREN);
    }
    setState(479);
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
  enterRule(_localctx, 82, qasm3Parser::RuleQuantumBlock);
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
    setState(481);
    match(qasm3Parser::LBRACE);
    setState(486);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__19)
      | (1ULL << qasm3Parser::T__20)
      | (1ULL << qasm3Parser::T__21)
      | (1ULL << qasm3Parser::T__22)
      | (1ULL << qasm3Parser::T__23)
      | (1ULL << qasm3Parser::T__24)
      | (1ULL << qasm3Parser::T__25)
      | (1ULL << qasm3Parser::T__26)
      | (1ULL << qasm3Parser::T__28))) != 0) || ((((_la - 76) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 76)) & ((1ULL << (qasm3Parser::T__75 - 76))
      | (1ULL << (qasm3Parser::T__76 - 76))
      | (1ULL << (qasm3Parser::T__87 - 76))
      | (1ULL << (qasm3Parser::T__88 - 76))
      | (1ULL << (qasm3Parser::T__90 - 76))
      | (1ULL << (qasm3Parser::T__91 - 76))
      | (1ULL << (qasm3Parser::Identifier - 76)))) != 0)) {
      setState(484);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case qasm3Parser::T__19:
        case qasm3Parser::T__20:
        case qasm3Parser::T__21:
        case qasm3Parser::T__22:
        case qasm3Parser::T__23:
        case qasm3Parser::T__24:
        case qasm3Parser::T__25:
        case qasm3Parser::T__26:
        case qasm3Parser::T__28:
        case qasm3Parser::T__87:
        case qasm3Parser::T__88:
        case qasm3Parser::T__90:
        case qasm3Parser::T__91:
        case qasm3Parser::Identifier: {
          setState(482);
          quantumStatement();
          break;
        }

        case qasm3Parser::T__75:
        case qasm3Parser::T__76: {
          setState(483);
          quantumLoop();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(488);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(489);
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
  enterRule(_localctx, 84, qasm3Parser::RuleQuantumLoop);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(491);
    loopSignature();
    setState(492);
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
  enterRule(_localctx, 86, qasm3Parser::RuleQuantumLoopBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(503);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__19:
      case qasm3Parser::T__20:
      case qasm3Parser::T__21:
      case qasm3Parser::T__22:
      case qasm3Parser::T__23:
      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26:
      case qasm3Parser::T__28:
      case qasm3Parser::T__87:
      case qasm3Parser::T__88:
      case qasm3Parser::T__90:
      case qasm3Parser::T__91:
      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(494);
        quantumStatement();
        break;
      }

      case qasm3Parser::LBRACE: {
        enterOuterAlt(_localctx, 2);
        setState(495);
        match(qasm3Parser::LBRACE);
        setState(499);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << qasm3Parser::T__19)
          | (1ULL << qasm3Parser::T__20)
          | (1ULL << qasm3Parser::T__21)
          | (1ULL << qasm3Parser::T__22)
          | (1ULL << qasm3Parser::T__23)
          | (1ULL << qasm3Parser::T__24)
          | (1ULL << qasm3Parser::T__25)
          | (1ULL << qasm3Parser::T__26)
          | (1ULL << qasm3Parser::T__28))) != 0) || ((((_la - 88) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 88)) & ((1ULL << (qasm3Parser::T__87 - 88))
          | (1ULL << (qasm3Parser::T__88 - 88))
          | (1ULL << (qasm3Parser::T__90 - 88))
          | (1ULL << (qasm3Parser::T__91 - 88))
          | (1ULL << (qasm3Parser::Identifier - 88)))) != 0)) {
          setState(496);
          quantumStatement();
          setState(501);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(502);
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
  enterRule(_localctx, 88, qasm3Parser::RuleQuantumStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(509);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__19:
      case qasm3Parser::T__20:
      case qasm3Parser::T__21:
      case qasm3Parser::T__22:
      case qasm3Parser::T__23:
      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26:
      case qasm3Parser::T__28:
      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(505);
        quantumInstruction();
        setState(506);
        match(qasm3Parser::SEMICOLON);
        break;
      }

      case qasm3Parser::T__87:
      case qasm3Parser::T__88:
      case qasm3Parser::T__90:
      case qasm3Parser::T__91: {
        enterOuterAlt(_localctx, 2);
        setState(508);
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
  enterRule(_localctx, 90, qasm3Parser::RuleQuantumInstruction);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(515);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__19:
      case qasm3Parser::T__20:
      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26:
      case qasm3Parser::T__28:
      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(511);
        quantumGateCall();
        break;
      }

      case qasm3Parser::T__21: {
        enterOuterAlt(_localctx, 2);
        setState(512);
        quantumPhase();
        break;
      }

      case qasm3Parser::T__22: {
        enterOuterAlt(_localctx, 3);
        setState(513);
        quantumMeasurement();
        break;
      }

      case qasm3Parser::T__23: {
        enterOuterAlt(_localctx, 4);
        setState(514);
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
  enterRule(_localctx, 92, qasm3Parser::RuleQuantumPhase);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(517);
    match(qasm3Parser::T__21);
    setState(518);
    match(qasm3Parser::LPAREN);
    setState(519);
    match(qasm3Parser::Identifier);
    setState(520);
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
  enterRule(_localctx, 94, qasm3Parser::RuleQuantumMeasurement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(522);
    match(qasm3Parser::T__22);
    setState(523);
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
  enterRule(_localctx, 96, qasm3Parser::RuleQuantumMeasurementAssignment);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(534);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__22: {
        enterOuterAlt(_localctx, 1);
        setState(525);
        quantumMeasurement();
        setState(528);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == qasm3Parser::ARROW) {
          setState(526);
          match(qasm3Parser::ARROW);
          setState(527);
          indexIdentifierList();
        }
        break;
      }

      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(530);
        indexIdentifierList();
        setState(531);
        match(qasm3Parser::EQUALS);
        setState(532);
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
  enterRule(_localctx, 98, qasm3Parser::RuleQuantumBarrier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(536);
    match(qasm3Parser::T__23);
    setState(537);
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
  enterRule(_localctx, 100, qasm3Parser::RuleQuantumGateModifier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(546);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__24: {
        setState(539);
        match(qasm3Parser::T__24);
        break;
      }

      case qasm3Parser::T__25: {
        setState(540);
        match(qasm3Parser::T__25);
        setState(541);
        match(qasm3Parser::LPAREN);
        setState(542);
        expression(0);
        setState(543);
        match(qasm3Parser::RPAREN);
        break;
      }

      case qasm3Parser::T__26: {
        setState(545);
        match(qasm3Parser::T__26);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(548);
    match(qasm3Parser::T__27);
   
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
  enterRule(_localctx, 102, qasm3Parser::RuleQuantumGateCall);
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
    setState(550);
    quantumGateName();
    setState(556);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(551);
      match(qasm3Parser::LPAREN);
      setState(553);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
        | (1ULL << qasm3Parser::T__6)
        | (1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__29)
        | (1ULL << qasm3Parser::T__30)
        | (1ULL << qasm3Parser::T__49)
        | (1ULL << qasm3Parser::T__50)
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
        | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 86) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 86)) & ((1ULL << (qasm3Parser::T__85 - 86))
        | (1ULL << (qasm3Parser::T__86 - 86))
        | (1ULL << (qasm3Parser::T__89 - 86))
        | (1ULL << (qasm3Parser::LPAREN - 86))
        | (1ULL << (qasm3Parser::MINUS - 86))
        | (1ULL << (qasm3Parser::Constant - 86))
        | (1ULL << (qasm3Parser::Integer - 86))
        | (1ULL << (qasm3Parser::Identifier - 86))
        | (1ULL << (qasm3Parser::RealNumber - 86))
        | (1ULL << (qasm3Parser::TimingLiteral - 86))
        | (1ULL << (qasm3Parser::StringLiteral - 86)))) != 0)) {
        setState(552);
        expressionList();
      }
      setState(555);
      match(qasm3Parser::RPAREN);
    }
    setState(558);
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
  enterRule(_localctx, 104, qasm3Parser::RuleQuantumGateName);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(567);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__19: {
        enterOuterAlt(_localctx, 1);
        setState(560);
        match(qasm3Parser::T__19);
        break;
      }

      case qasm3Parser::T__20: {
        enterOuterAlt(_localctx, 2);
        setState(561);
        match(qasm3Parser::T__20);
        break;
      }

      case qasm3Parser::T__28: {
        enterOuterAlt(_localctx, 3);
        setState(562);
        match(qasm3Parser::T__28);
        break;
      }

      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 4);
        setState(563);
        match(qasm3Parser::Identifier);
        break;
      }

      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26: {
        enterOuterAlt(_localctx, 5);
        setState(564);
        quantumGateModifier();
        setState(565);
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
  enterRule(_localctx, 106, qasm3Parser::RuleUnaryOperator);
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
    setState(569);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__29

    || _la == qasm3Parser::T__30)) {
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
  enterRule(_localctx, 108, qasm3Parser::RuleRelationalOperator);
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
    setState(571);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__31)
      | (1ULL << qasm3Parser::T__32)
      | (1ULL << qasm3Parser::T__33)
      | (1ULL << qasm3Parser::T__34)
      | (1ULL << qasm3Parser::T__35)
      | (1ULL << qasm3Parser::T__36))) != 0))) {
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
  enterRule(_localctx, 110, qasm3Parser::RuleLogicalOperator);
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
    setState(573);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__17

    || _la == qasm3Parser::T__37)) {
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
  enterRule(_localctx, 112, qasm3Parser::RuleExpressionStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(575);
    expression(0);
    setState(576);
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
  size_t startState = 114;
  enterRecursionRule(_localctx, 114, qasm3Parser::RuleExpression, precedence);

    

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
    setState(582);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 44, _ctx)) {
    case 1: {
      setState(579);
      expressionTerminator(0);
      break;
    }

    case 2: {
      setState(580);
      unaryExpression();
      break;
    }

    case 3: {
      setState(581);
      xOrExpression(0);
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(589);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 45, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleExpression);
        setState(584);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(585);
        match(qasm3Parser::T__38);
        setState(586);
        xOrExpression(0); 
      }
      setState(591);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 45, _ctx);
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
  size_t startState = 116;
  enterRecursionRule(_localctx, 116, qasm3Parser::RuleXOrExpression, precedence);

    

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
    setState(593);
    bitAndExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(600);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 46, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<XOrExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleXOrExpression);
        setState(595);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(596);
        match(qasm3Parser::T__39);
        setState(597);
        bitAndExpression(0); 
      }
      setState(602);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 46, _ctx);
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
  size_t startState = 118;
  enterRecursionRule(_localctx, 118, qasm3Parser::RuleBitAndExpression, precedence);

    

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
    setState(604);
    bitShiftExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(611);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 47, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<BitAndExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleBitAndExpression);
        setState(606);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(607);
        match(qasm3Parser::T__40);
        setState(608);
        bitShiftExpression(0); 
      }
      setState(613);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 47, _ctx);
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
  size_t startState = 120;
  enterRecursionRule(_localctx, 120, qasm3Parser::RuleBitShiftExpression, precedence);

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
    setState(615);
    additiveExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(622);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 48, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<BitShiftExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleBitShiftExpression);
        setState(617);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(618);
        _la = _input->LA(1);
        if (!(_la == qasm3Parser::T__41

        || _la == qasm3Parser::T__42)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(619);
        additiveExpression(0); 
      }
      setState(624);
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
  size_t startState = 122;
  enterRecursionRule(_localctx, 122, qasm3Parser::RuleAdditiveExpression, precedence);

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
    setState(626);
    multiplicativeExpression(0);
    _ctx->stop = _input->LT(-1);
    setState(633);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 49, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<AdditiveExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleAdditiveExpression);
        setState(628);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(629);
        dynamic_cast<AdditiveExpressionContext *>(_localctx)->binary_op = _input->LT(1);
        _la = _input->LA(1);
        if (!(_la == qasm3Parser::T__43 || _la == qasm3Parser::MINUS)) {
          dynamic_cast<AdditiveExpressionContext *>(_localctx)->binary_op = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(630);
        multiplicativeExpression(0); 
      }
      setState(635);
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
  size_t startState = 124;
  enterRecursionRule(_localctx, 124, qasm3Parser::RuleMultiplicativeExpression, precedence);

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
    setState(639);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__5:
      case qasm3Parser::T__6:
      case qasm3Parser::T__7:
      case qasm3Parser::T__8:
      case qasm3Parser::T__9:
      case qasm3Parser::T__10:
      case qasm3Parser::T__11:
      case qasm3Parser::T__12:
      case qasm3Parser::T__13:
      case qasm3Parser::T__14:
      case qasm3Parser::T__49:
      case qasm3Parser::T__50:
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
      case qasm3Parser::T__85:
      case qasm3Parser::T__86:
      case qasm3Parser::T__89:
      case qasm3Parser::LPAREN:
      case qasm3Parser::MINUS:
      case qasm3Parser::Constant:
      case qasm3Parser::Integer:
      case qasm3Parser::Identifier:
      case qasm3Parser::RealNumber:
      case qasm3Parser::TimingLiteral:
      case qasm3Parser::StringLiteral: {
        setState(637);
        expressionTerminator(0);
        break;
      }

      case qasm3Parser::T__29:
      case qasm3Parser::T__30: {
        setState(638);
        unaryExpression();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(649);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<MultiplicativeExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleMultiplicativeExpression);
        setState(641);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(642);
        dynamic_cast<MultiplicativeExpressionContext *>(_localctx)->binary_op = _input->LT(1);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << qasm3Parser::T__44)
          | (1ULL << qasm3Parser::T__45)
          | (1ULL << qasm3Parser::T__46))) != 0))) {
          dynamic_cast<MultiplicativeExpressionContext *>(_localctx)->binary_op = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(645);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case qasm3Parser::T__5:
          case qasm3Parser::T__6:
          case qasm3Parser::T__7:
          case qasm3Parser::T__8:
          case qasm3Parser::T__9:
          case qasm3Parser::T__10:
          case qasm3Parser::T__11:
          case qasm3Parser::T__12:
          case qasm3Parser::T__13:
          case qasm3Parser::T__14:
          case qasm3Parser::T__49:
          case qasm3Parser::T__50:
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
          case qasm3Parser::T__85:
          case qasm3Parser::T__86:
          case qasm3Parser::T__89:
          case qasm3Parser::LPAREN:
          case qasm3Parser::MINUS:
          case qasm3Parser::Constant:
          case qasm3Parser::Integer:
          case qasm3Parser::Identifier:
          case qasm3Parser::RealNumber:
          case qasm3Parser::TimingLiteral:
          case qasm3Parser::StringLiteral: {
            setState(643);
            expressionTerminator(0);
            break;
          }

          case qasm3Parser::T__29:
          case qasm3Parser::T__30: {
            setState(644);
            unaryExpression();
            break;
          }

        default:
          throw NoViableAltException(this);
        } 
      }
      setState(651);
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
  enterRule(_localctx, 126, qasm3Parser::RuleUnaryExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(652);
    unaryOperator();
    setState(653);
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
  size_t startState = 128;
  enterRecursionRule(_localctx, 128, qasm3Parser::RuleExpressionTerminator, precedence);

    

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
    setState(671);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 53, _ctx)) {
    case 1: {
      setState(656);
      match(qasm3Parser::Constant);
      break;
    }

    case 2: {
      setState(657);
      match(qasm3Parser::Integer);
      break;
    }

    case 3: {
      setState(658);
      match(qasm3Parser::RealNumber);
      break;
    }

    case 4: {
      setState(659);
      match(qasm3Parser::Identifier);
      break;
    }

    case 5: {
      setState(660);
      match(qasm3Parser::StringLiteral);
      break;
    }

    case 6: {
      setState(661);
      builtInCall();
      break;
    }

    case 7: {
      setState(662);
      kernelCall();
      break;
    }

    case 8: {
      setState(663);
      subroutineCall();
      break;
    }

    case 9: {
      setState(664);
      timingTerminator();
      break;
    }

    case 10: {
      setState(665);
      match(qasm3Parser::MINUS);
      setState(666);
      expressionTerminator(4);
      break;
    }

    case 11: {
      setState(667);
      match(qasm3Parser::LPAREN);
      setState(668);
      expression(0);
      setState(669);
      match(qasm3Parser::RPAREN);
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(682);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 55, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(680);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 54, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExpressionTerminatorContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpressionTerminator);
          setState(673);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(674);
          match(qasm3Parser::LBRACKET);
          setState(675);
          expression(0);
          setState(676);
          match(qasm3Parser::RBRACKET);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExpressionTerminatorContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpressionTerminator);
          setState(678);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(679);
          incrementor();
          break;
        }

        default:
          break;
        } 
      }
      setState(684);
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
  enterRule(_localctx, 130, qasm3Parser::RuleIncrementor);
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
    setState(685);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__47

    || _la == qasm3Parser::T__48)) {
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
  enterRule(_localctx, 132, qasm3Parser::RuleBuiltInCall);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(689);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__49:
      case qasm3Parser::T__50:
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
      case qasm3Parser::T__61: {
        setState(687);
        builtInMath();
        break;
      }

      case qasm3Parser::T__5:
      case qasm3Parser::T__6:
      case qasm3Parser::T__7:
      case qasm3Parser::T__8:
      case qasm3Parser::T__9:
      case qasm3Parser::T__10:
      case qasm3Parser::T__11:
      case qasm3Parser::T__12:
      case qasm3Parser::T__13:
      case qasm3Parser::T__14:
      case qasm3Parser::T__85:
      case qasm3Parser::T__86: {
        setState(688);
        castOperator();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(691);
    match(qasm3Parser::LPAREN);
    setState(692);
    expressionList();
    setState(693);
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
  enterRule(_localctx, 134, qasm3Parser::RuleBuiltInMath);
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
    setState(695);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__49)
      | (1ULL << qasm3Parser::T__50)
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
      | (1ULL << qasm3Parser::T__61))) != 0))) {
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
  enterRule(_localctx, 136, qasm3Parser::RuleCastOperator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(697);
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
  enterRule(_localctx, 138, qasm3Parser::RuleExpressionList);

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
    setState(704);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 57, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(699);
        expression(0);
        setState(700);
        match(qasm3Parser::COMMA); 
      }
      setState(706);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 57, _ctx);
    }
    setState(707);
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
  size_t startState = 140;
  enterRecursionRule(_localctx, 140, qasm3Parser::RuleBooleanExpression, precedence);

    

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
    setState(712);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 58, _ctx)) {
    case 1: {
      setState(710);
      membershipTest();
      break;
    }

    case 2: {
      setState(711);
      comparsionExpression();
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(720);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 59, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<BooleanExpressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleBooleanExpression);
        setState(714);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(715);
        logicalOperator();
        setState(716);
        comparsionExpression(); 
      }
      setState(722);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 59, _ctx);
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
  enterRule(_localctx, 142, qasm3Parser::RuleComparsionExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(728);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 60, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(723);
      expression(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(724);
      expression(0);
      setState(725);
      relationalOperator();
      setState(726);
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
  enterRule(_localctx, 144, qasm3Parser::RuleEqualsExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(730);
    match(qasm3Parser::EQUALS);
    setState(731);
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
  enterRule(_localctx, 146, qasm3Parser::RuleAssignmentOperator);
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
    setState(733);
    _la = _input->LA(1);
    if (!(((((_la - 63) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 63)) & ((1ULL << (qasm3Parser::T__62 - 63))
      | (1ULL << (qasm3Parser::T__63 - 63))
      | (1ULL << (qasm3Parser::T__64 - 63))
      | (1ULL << (qasm3Parser::T__65 - 63))
      | (1ULL << (qasm3Parser::T__66 - 63))
      | (1ULL << (qasm3Parser::T__67 - 63))
      | (1ULL << (qasm3Parser::T__68 - 63))
      | (1ULL << (qasm3Parser::T__69 - 63))
      | (1ULL << (qasm3Parser::T__70 - 63))
      | (1ULL << (qasm3Parser::T__71 - 63))
      | (1ULL << (qasm3Parser::EQUALS - 63)))) != 0))) {
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
  enterRule(_localctx, 148, qasm3Parser::RuleEqualsAssignmentList);

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
    setState(741);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 61, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(735);
        match(qasm3Parser::Identifier);
        setState(736);
        equalsExpression();
        setState(737);
        match(qasm3Parser::COMMA); 
      }
      setState(743);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 61, _ctx);
    }
    setState(744);
    match(qasm3Parser::Identifier);
    setState(745);
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
  enterRule(_localctx, 150, qasm3Parser::RuleMembershipTest);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(747);
    match(qasm3Parser::Identifier);
    setState(748);
    match(qasm3Parser::T__72);
    setState(749);
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
  enterRule(_localctx, 152, qasm3Parser::RuleSetDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(757);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::LBRACE: {
        enterOuterAlt(_localctx, 1);
        setState(751);
        match(qasm3Parser::LBRACE);
        setState(752);
        expressionList();
        setState(753);
        match(qasm3Parser::RBRACE);
        break;
      }

      case qasm3Parser::LBRACKET: {
        enterOuterAlt(_localctx, 2);
        setState(755);
        rangeDefinition();
        break;
      }

      case qasm3Parser::Identifier: {
        enterOuterAlt(_localctx, 3);
        setState(756);
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
  enterRule(_localctx, 154, qasm3Parser::RuleProgramBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(768);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__2:
      case qasm3Parser::T__5:
      case qasm3Parser::T__6:
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
      case qasm3Parser::T__19:
      case qasm3Parser::T__20:
      case qasm3Parser::T__21:
      case qasm3Parser::T__22:
      case qasm3Parser::T__23:
      case qasm3Parser::T__24:
      case qasm3Parser::T__25:
      case qasm3Parser::T__26:
      case qasm3Parser::T__28:
      case qasm3Parser::T__29:
      case qasm3Parser::T__30:
      case qasm3Parser::T__49:
      case qasm3Parser::T__50:
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
      case qasm3Parser::T__73:
      case qasm3Parser::T__75:
      case qasm3Parser::T__76:
      case qasm3Parser::T__78:
      case qasm3Parser::T__79:
      case qasm3Parser::T__80:
      case qasm3Parser::T__83:
      case qasm3Parser::T__85:
      case qasm3Parser::T__86:
      case qasm3Parser::T__87:
      case qasm3Parser::T__88:
      case qasm3Parser::T__89:
      case qasm3Parser::T__90:
      case qasm3Parser::T__91:
      case qasm3Parser::LPAREN:
      case qasm3Parser::MINUS:
      case qasm3Parser::Constant:
      case qasm3Parser::Integer:
      case qasm3Parser::Identifier:
      case qasm3Parser::RealNumber:
      case qasm3Parser::TimingLiteral:
      case qasm3Parser::StringLiteral: {
        enterOuterAlt(_localctx, 1);
        setState(759);
        statement();
        break;
      }

      case qasm3Parser::LBRACE: {
        enterOuterAlt(_localctx, 2);
        setState(760);
        match(qasm3Parser::LBRACE);
        setState(764);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << qasm3Parser::T__2)
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
          | (1ULL << qasm3Parser::T__19)
          | (1ULL << qasm3Parser::T__20)
          | (1ULL << qasm3Parser::T__21)
          | (1ULL << qasm3Parser::T__22)
          | (1ULL << qasm3Parser::T__23)
          | (1ULL << qasm3Parser::T__24)
          | (1ULL << qasm3Parser::T__25)
          | (1ULL << qasm3Parser::T__26)
          | (1ULL << qasm3Parser::T__28)
          | (1ULL << qasm3Parser::T__29)
          | (1ULL << qasm3Parser::T__30)
          | (1ULL << qasm3Parser::T__49)
          | (1ULL << qasm3Parser::T__50)
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
          | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 74) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 74)) & ((1ULL << (qasm3Parser::T__73 - 74))
          | (1ULL << (qasm3Parser::T__75 - 74))
          | (1ULL << (qasm3Parser::T__76 - 74))
          | (1ULL << (qasm3Parser::T__78 - 74))
          | (1ULL << (qasm3Parser::T__79 - 74))
          | (1ULL << (qasm3Parser::T__80 - 74))
          | (1ULL << (qasm3Parser::T__83 - 74))
          | (1ULL << (qasm3Parser::T__85 - 74))
          | (1ULL << (qasm3Parser::T__86 - 74))
          | (1ULL << (qasm3Parser::T__87 - 74))
          | (1ULL << (qasm3Parser::T__88 - 74))
          | (1ULL << (qasm3Parser::T__89 - 74))
          | (1ULL << (qasm3Parser::T__90 - 74))
          | (1ULL << (qasm3Parser::T__91 - 74))
          | (1ULL << (qasm3Parser::LPAREN - 74))
          | (1ULL << (qasm3Parser::MINUS - 74))
          | (1ULL << (qasm3Parser::Constant - 74))
          | (1ULL << (qasm3Parser::Integer - 74))
          | (1ULL << (qasm3Parser::Identifier - 74))
          | (1ULL << (qasm3Parser::RealNumber - 74))
          | (1ULL << (qasm3Parser::TimingLiteral - 74))
          | (1ULL << (qasm3Parser::StringLiteral - 74)))) != 0)) {
          setState(761);
          statement();
          setState(766);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(767);
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
  enterRule(_localctx, 156, qasm3Parser::RuleBranchingStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(770);
    match(qasm3Parser::T__73);
    setState(771);
    match(qasm3Parser::LPAREN);
    setState(772);
    booleanExpression(0);
    setState(773);
    match(qasm3Parser::RPAREN);
    setState(774);
    programBlock();
    setState(777);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 65, _ctx)) {
    case 1: {
      setState(775);
      match(qasm3Parser::T__74);
      setState(776);
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
  enterRule(_localctx, 158, qasm3Parser::RuleLoopSignature);

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
      case qasm3Parser::T__75: {
        enterOuterAlt(_localctx, 1);
        setState(779);
        match(qasm3Parser::T__75);
        setState(780);
        membershipTest();
        break;
      }

      case qasm3Parser::T__76: {
        enterOuterAlt(_localctx, 2);
        setState(781);
        match(qasm3Parser::T__76);
        setState(782);
        match(qasm3Parser::LPAREN);
        setState(783);
        booleanExpression(0);
        setState(784);
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
  enterRule(_localctx, 160, qasm3Parser::RuleLoopStatement);

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
    loopSignature();
    setState(789);
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
  enterRule(_localctx, 162, qasm3Parser::RuleCLikeLoopStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(819);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 67, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(791);
      match(qasm3Parser::T__75);
      setState(792);
      match(qasm3Parser::LPAREN);
      setState(793);
      classicalType();
      setState(794);
      match(qasm3Parser::Identifier);
      setState(795);
      equalsExpression();
      setState(796);
      match(qasm3Parser::SEMICOLON);
      setState(797);
      booleanExpression(0);
      setState(798);
      match(qasm3Parser::SEMICOLON);
      setState(799);
      expression(0);
      setState(800);
      match(qasm3Parser::RPAREN);
      setState(801);
      match(qasm3Parser::LBRACE);
      setState(802);
      programBlock();
      setState(803);
      match(qasm3Parser::RBRACE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(805);
      match(qasm3Parser::T__75);
      setState(806);
      match(qasm3Parser::LPAREN);
      setState(807);
      classicalType();
      setState(808);
      match(qasm3Parser::Identifier);
      setState(809);
      match(qasm3Parser::COLON);
      setState(810);
      match(qasm3Parser::T__77);
      setState(811);
      match(qasm3Parser::LPAREN);
      setState(812);
      expression(0);
      setState(813);
      match(qasm3Parser::RPAREN);
      setState(814);
      match(qasm3Parser::RPAREN);
      setState(815);
      match(qasm3Parser::LBRACE);
      setState(816);
      programBlock();
      setState(817);
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
  enterRule(_localctx, 164, qasm3Parser::RuleControlDirectiveStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(821);
    controlDirective();
    setState(822);
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
  enterRule(_localctx, 166, qasm3Parser::RuleControlDirective);
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
    setState(824);
    _la = _input->LA(1);
    if (!(((((_la - 79) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 79)) & ((1ULL << (qasm3Parser::T__78 - 79))
      | (1ULL << (qasm3Parser::T__79 - 79))
      | (1ULL << (qasm3Parser::T__80 - 79)))) != 0))) {
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
  enterRule(_localctx, 168, qasm3Parser::RuleKernelDeclaration);
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
    setState(826);
    match(qasm3Parser::T__81);
    setState(827);
    match(qasm3Parser::Identifier);
    setState(833);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(828);
      match(qasm3Parser::LPAREN);
      setState(830);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
        | (1ULL << qasm3Parser::T__6)
        | (1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14))) != 0) || _la == qasm3Parser::T__85

      || _la == qasm3Parser::T__86) {
        setState(829);
        classicalTypeList();
      }
      setState(832);
      match(qasm3Parser::RPAREN);
    }
    setState(836);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::ARROW) {
      setState(835);
      returnSignature();
    }
    setState(839);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
      | (1ULL << qasm3Parser::T__6)
      | (1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14))) != 0) || _la == qasm3Parser::T__85

    || _la == qasm3Parser::T__86) {
      setState(838);
      classicalType();
    }
    setState(841);
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
  enterRule(_localctx, 170, qasm3Parser::RuleKernelCall);
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
    setState(843);
    match(qasm3Parser::Identifier);
    setState(844);
    match(qasm3Parser::LPAREN);
    setState(846);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
      | (1ULL << qasm3Parser::T__6)
      | (1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__29)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__49)
      | (1ULL << qasm3Parser::T__50)
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
      | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 86) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 86)) & ((1ULL << (qasm3Parser::T__85 - 86))
      | (1ULL << (qasm3Parser::T__86 - 86))
      | (1ULL << (qasm3Parser::T__89 - 86))
      | (1ULL << (qasm3Parser::LPAREN - 86))
      | (1ULL << (qasm3Parser::MINUS - 86))
      | (1ULL << (qasm3Parser::Constant - 86))
      | (1ULL << (qasm3Parser::Integer - 86))
      | (1ULL << (qasm3Parser::Identifier - 86))
      | (1ULL << (qasm3Parser::RealNumber - 86))
      | (1ULL << (qasm3Parser::TimingLiteral - 86))
      | (1ULL << (qasm3Parser::StringLiteral - 86)))) != 0)) {
      setState(845);
      expressionList();
    }
    setState(848);
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
  enterRule(_localctx, 172, qasm3Parser::RuleSubroutineDefinition);
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
    setState(850);
    match(qasm3Parser::T__82);
    setState(851);
    match(qasm3Parser::Identifier);
    setState(857);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(852);
      match(qasm3Parser::LPAREN);
      setState(854);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
        | (1ULL << qasm3Parser::T__6)
        | (1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14))) != 0) || _la == qasm3Parser::T__85

      || _la == qasm3Parser::T__86) {
        setState(853);
        classicalArgumentList();
      }
      setState(856);
      match(qasm3Parser::RPAREN);
    }
    setState(860);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::T__3

    || _la == qasm3Parser::T__4) {
      setState(859);
      quantumArgumentList();
    }
    setState(863);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::ARROW) {
      setState(862);
      returnSignature();
    }
    setState(865);
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
  enterRule(_localctx, 174, qasm3Parser::RuleReturnStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(867);
    match(qasm3Parser::T__83);
    setState(868);
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
  enterRule(_localctx, 176, qasm3Parser::RuleSubroutineBlock);
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
    setState(870);
    match(qasm3Parser::LBRACE);
    setState(874);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 77, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(871);
        statement(); 
      }
      setState(876);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 77, _ctx);
    }
    setState(878);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::T__83) {
      setState(877);
      returnStatement();
    }
    setState(880);
    match(qasm3Parser::RBRACE);
   
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
  enterRule(_localctx, 178, qasm3Parser::RuleSubroutineCall);
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
    setState(882);
    match(qasm3Parser::Identifier);

    setState(883);
    match(qasm3Parser::LPAREN);
    setState(885);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
      | (1ULL << qasm3Parser::T__6)
      | (1ULL << qasm3Parser::T__7)
      | (1ULL << qasm3Parser::T__8)
      | (1ULL << qasm3Parser::T__9)
      | (1ULL << qasm3Parser::T__10)
      | (1ULL << qasm3Parser::T__11)
      | (1ULL << qasm3Parser::T__12)
      | (1ULL << qasm3Parser::T__13)
      | (1ULL << qasm3Parser::T__14)
      | (1ULL << qasm3Parser::T__29)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__49)
      | (1ULL << qasm3Parser::T__50)
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
      | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 86) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 86)) & ((1ULL << (qasm3Parser::T__85 - 86))
      | (1ULL << (qasm3Parser::T__86 - 86))
      | (1ULL << (qasm3Parser::T__89 - 86))
      | (1ULL << (qasm3Parser::LPAREN - 86))
      | (1ULL << (qasm3Parser::MINUS - 86))
      | (1ULL << (qasm3Parser::Constant - 86))
      | (1ULL << (qasm3Parser::Integer - 86))
      | (1ULL << (qasm3Parser::Identifier - 86))
      | (1ULL << (qasm3Parser::RealNumber - 86))
      | (1ULL << (qasm3Parser::TimingLiteral - 86))
      | (1ULL << (qasm3Parser::StringLiteral - 86)))) != 0)) {
      setState(884);
      expressionList();
    }
    setState(887);
    match(qasm3Parser::RPAREN);
    setState(889);
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
  enterRule(_localctx, 180, qasm3Parser::RulePragma);
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
    setState(891);
    match(qasm3Parser::T__84);
    setState(892);
    match(qasm3Parser::LBRACE);
    setState(896);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasm3Parser::T__2)
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
      | (1ULL << qasm3Parser::T__19)
      | (1ULL << qasm3Parser::T__20)
      | (1ULL << qasm3Parser::T__21)
      | (1ULL << qasm3Parser::T__22)
      | (1ULL << qasm3Parser::T__23)
      | (1ULL << qasm3Parser::T__24)
      | (1ULL << qasm3Parser::T__25)
      | (1ULL << qasm3Parser::T__26)
      | (1ULL << qasm3Parser::T__28)
      | (1ULL << qasm3Parser::T__29)
      | (1ULL << qasm3Parser::T__30)
      | (1ULL << qasm3Parser::T__49)
      | (1ULL << qasm3Parser::T__50)
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
      | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 74) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 74)) & ((1ULL << (qasm3Parser::T__73 - 74))
      | (1ULL << (qasm3Parser::T__75 - 74))
      | (1ULL << (qasm3Parser::T__76 - 74))
      | (1ULL << (qasm3Parser::T__78 - 74))
      | (1ULL << (qasm3Parser::T__79 - 74))
      | (1ULL << (qasm3Parser::T__80 - 74))
      | (1ULL << (qasm3Parser::T__83 - 74))
      | (1ULL << (qasm3Parser::T__85 - 74))
      | (1ULL << (qasm3Parser::T__86 - 74))
      | (1ULL << (qasm3Parser::T__87 - 74))
      | (1ULL << (qasm3Parser::T__88 - 74))
      | (1ULL << (qasm3Parser::T__89 - 74))
      | (1ULL << (qasm3Parser::T__90 - 74))
      | (1ULL << (qasm3Parser::T__91 - 74))
      | (1ULL << (qasm3Parser::LPAREN - 74))
      | (1ULL << (qasm3Parser::MINUS - 74))
      | (1ULL << (qasm3Parser::Constant - 74))
      | (1ULL << (qasm3Parser::Integer - 74))
      | (1ULL << (qasm3Parser::Identifier - 74))
      | (1ULL << (qasm3Parser::RealNumber - 74))
      | (1ULL << (qasm3Parser::TimingLiteral - 74))
      | (1ULL << (qasm3Parser::StringLiteral - 74)))) != 0)) {
      setState(893);
      statement();
      setState(898);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(899);
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
  enterRule(_localctx, 182, qasm3Parser::RuleTimingType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(906);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__85: {
        enterOuterAlt(_localctx, 1);
        setState(901);
        match(qasm3Parser::T__85);
        break;
      }

      case qasm3Parser::T__86: {
        enterOuterAlt(_localctx, 2);
        setState(902);
        match(qasm3Parser::T__86);
        setState(904);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == qasm3Parser::Integer) {
          setState(903);
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
  enterRule(_localctx, 184, qasm3Parser::RuleTimingBox);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(914);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__87: {
        enterOuterAlt(_localctx, 1);
        setState(908);
        match(qasm3Parser::T__87);
        setState(909);
        match(qasm3Parser::Identifier);
        setState(910);
        quantumBlock();
        break;
      }

      case qasm3Parser::T__88: {
        enterOuterAlt(_localctx, 2);
        setState(911);
        match(qasm3Parser::T__88);
        setState(912);
        match(qasm3Parser::TimingLiteral);
        setState(913);
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
  enterRule(_localctx, 186, qasm3Parser::RuleTimingTerminator);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(918);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__61:
      case qasm3Parser::TimingLiteral: {
        enterOuterAlt(_localctx, 1);
        setState(916);
        timingIdentifier();
        break;
      }

      case qasm3Parser::T__89: {
        enterOuterAlt(_localctx, 2);
        setState(917);
        match(qasm3Parser::T__89);
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
  enterRule(_localctx, 188, qasm3Parser::RuleTimingIdentifier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(928);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::TimingLiteral: {
        enterOuterAlt(_localctx, 1);
        setState(920);
        match(qasm3Parser::TimingLiteral);
        break;
      }

      case qasm3Parser::T__61: {
        enterOuterAlt(_localctx, 2);
        setState(921);
        match(qasm3Parser::T__61);
        setState(922);
        match(qasm3Parser::LPAREN);
        setState(925);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case qasm3Parser::Identifier: {
            setState(923);
            match(qasm3Parser::Identifier);
            break;
          }

          case qasm3Parser::LBRACE: {
            setState(924);
            quantumBlock();
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        setState(927);
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
  enterRule(_localctx, 190, qasm3Parser::RuleTimingInstructionName);
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
    setState(930);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__90

    || _la == qasm3Parser::T__91)) {
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
  enterRule(_localctx, 192, qasm3Parser::RuleTimingInstruction);
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
    setState(932);
    timingInstructionName();
    setState(938);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(933);
      match(qasm3Parser::LPAREN);
      setState(935);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
        | (1ULL << qasm3Parser::T__6)
        | (1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__29)
        | (1ULL << qasm3Parser::T__30)
        | (1ULL << qasm3Parser::T__49)
        | (1ULL << qasm3Parser::T__50)
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
        | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 86) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 86)) & ((1ULL << (qasm3Parser::T__85 - 86))
        | (1ULL << (qasm3Parser::T__86 - 86))
        | (1ULL << (qasm3Parser::T__89 - 86))
        | (1ULL << (qasm3Parser::LPAREN - 86))
        | (1ULL << (qasm3Parser::MINUS - 86))
        | (1ULL << (qasm3Parser::Constant - 86))
        | (1ULL << (qasm3Parser::Integer - 86))
        | (1ULL << (qasm3Parser::Identifier - 86))
        | (1ULL << (qasm3Parser::RealNumber - 86))
        | (1ULL << (qasm3Parser::TimingLiteral - 86))
        | (1ULL << (qasm3Parser::StringLiteral - 86)))) != 0)) {
        setState(934);
        expressionList();
      }
      setState(937);
      match(qasm3Parser::RPAREN);
    }
    setState(940);
    designator();
    setState(941);
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
  enterRule(_localctx, 194, qasm3Parser::RuleTimingStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(947);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__90:
      case qasm3Parser::T__91: {
        enterOuterAlt(_localctx, 1);
        setState(943);
        timingInstruction();
        setState(944);
        match(qasm3Parser::SEMICOLON);
        break;
      }

      case qasm3Parser::T__87:
      case qasm3Parser::T__88: {
        enterOuterAlt(_localctx, 2);
        setState(946);
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
  enterRule(_localctx, 196, qasm3Parser::RuleCalibration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(951);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasm3Parser::T__92: {
        enterOuterAlt(_localctx, 1);
        setState(949);
        calibrationGrammarDeclaration();
        break;
      }

      case qasm3Parser::T__93: {
        enterOuterAlt(_localctx, 2);
        setState(950);
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
  enterRule(_localctx, 198, qasm3Parser::RuleCalibrationGrammarDeclaration);

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
    match(qasm3Parser::T__92);
    setState(954);
    calibrationGrammar();
    setState(955);
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
  enterRule(_localctx, 200, qasm3Parser::RuleCalibrationDefinition);
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
    setState(957);
    match(qasm3Parser::T__93);
    setState(958);
    match(qasm3Parser::Identifier);
    setState(964);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::LPAREN) {
      setState(959);
      match(qasm3Parser::LPAREN);
      setState(961);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasm3Parser::T__5)
        | (1ULL << qasm3Parser::T__6)
        | (1ULL << qasm3Parser::T__7)
        | (1ULL << qasm3Parser::T__8)
        | (1ULL << qasm3Parser::T__9)
        | (1ULL << qasm3Parser::T__10)
        | (1ULL << qasm3Parser::T__11)
        | (1ULL << qasm3Parser::T__12)
        | (1ULL << qasm3Parser::T__13)
        | (1ULL << qasm3Parser::T__14)
        | (1ULL << qasm3Parser::T__29)
        | (1ULL << qasm3Parser::T__30)
        | (1ULL << qasm3Parser::T__49)
        | (1ULL << qasm3Parser::T__50)
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
        | (1ULL << qasm3Parser::T__61))) != 0) || ((((_la - 86) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 86)) & ((1ULL << (qasm3Parser::T__85 - 86))
        | (1ULL << (qasm3Parser::T__86 - 86))
        | (1ULL << (qasm3Parser::T__89 - 86))
        | (1ULL << (qasm3Parser::LPAREN - 86))
        | (1ULL << (qasm3Parser::MINUS - 86))
        | (1ULL << (qasm3Parser::Constant - 86))
        | (1ULL << (qasm3Parser::Integer - 86))
        | (1ULL << (qasm3Parser::Identifier - 86))
        | (1ULL << (qasm3Parser::RealNumber - 86))
        | (1ULL << (qasm3Parser::TimingLiteral - 86))
        | (1ULL << (qasm3Parser::StringLiteral - 86)))) != 0)) {
        setState(960);
        calibrationArgumentList();
      }
      setState(963);
      match(qasm3Parser::RPAREN);
    }
    setState(966);
    identifierList();
    setState(968);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == qasm3Parser::ARROW) {
      setState(967);
      returnSignature();
    }
    setState(970);
    match(qasm3Parser::LBRACE);
    setState(974);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 94, _ctx);
    while (alt != 1 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1 + 1) {
        setState(971);
        matchWildcard(); 
      }
      setState(976);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 94, _ctx);
    }
    setState(977);
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
  enterRule(_localctx, 202, qasm3Parser::RuleCalibrationGrammar);
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
    setState(979);
    _la = _input->LA(1);
    if (!(_la == qasm3Parser::T__94

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
  enterRule(_localctx, 204, qasm3Parser::RuleCalibrationArgumentList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(983);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 95, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(981);
      classicalArgumentList();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(982);
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
    case 35: return indexIdentifierSempred(dynamic_cast<IndexIdentifierContext *>(context), predicateIndex);
    case 57: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);
    case 58: return xOrExpressionSempred(dynamic_cast<XOrExpressionContext *>(context), predicateIndex);
    case 59: return bitAndExpressionSempred(dynamic_cast<BitAndExpressionContext *>(context), predicateIndex);
    case 60: return bitShiftExpressionSempred(dynamic_cast<BitShiftExpressionContext *>(context), predicateIndex);
    case 61: return additiveExpressionSempred(dynamic_cast<AdditiveExpressionContext *>(context), predicateIndex);
    case 62: return multiplicativeExpressionSempred(dynamic_cast<MultiplicativeExpressionContext *>(context), predicateIndex);
    case 64: return expressionTerminatorSempred(dynamic_cast<ExpressionTerminatorContext *>(context), predicateIndex);
    case 70: return booleanExpressionSempred(dynamic_cast<BooleanExpressionContext *>(context), predicateIndex);

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
  "qcor_test_statement", "quantumDeclarationStatement", "classicalDeclarationStatement", 
  "classicalAssignment", "assignmentStatement", "returnSignature", "designator", 
  "doubleDesignator", "identifierList", "association", "quantumType", "quantumDeclaration", 
  "quantumArgument", "quantumArgumentList", "bitType", "singleDesignatorType", 
  "doubleDesignatorType", "noDesignatorType", "classicalType", "constantDeclaration", 
  "singleDesignatorDeclaration", "doubleDesignatorDeclaration", "noDesignatorDeclaration", 
  "bitDeclaration", "classicalDeclaration", "classicalTypeList", "classicalArgument", 
  "classicalArgumentList", "aliasStatement", "indexIdentifier", "indexIdentifierList", 
  "indexEqualsAssignmentList", "rangeDefinition", "quantumGateDefinition", 
  "quantumGateSignature", "quantumBlock", "quantumLoop", "quantumLoopBlock", 
  "quantumStatement", "quantumInstruction", "quantumPhase", "quantumMeasurement", 
  "quantumMeasurementAssignment", "quantumBarrier", "quantumGateModifier", 
  "quantumGateCall", "quantumGateName", "unaryOperator", "relationalOperator", 
  "logicalOperator", "expressionStatement", "expression", "xOrExpression", 
  "bitAndExpression", "bitShiftExpression", "additiveExpression", "multiplicativeExpression", 
  "unaryExpression", "expressionTerminator", "incrementor", "builtInCall", 
  "builtInMath", "castOperator", "expressionList", "booleanExpression", 
  "comparsionExpression", "equalsExpression", "assignmentOperator", "equalsAssignmentList", 
  "membershipTest", "setDeclaration", "programBlock", "branchingStatement", 
  "loopSignature", "loopStatement", "cLikeLoopStatement", "controlDirectiveStatement", 
  "controlDirective", "kernelDeclaration", "kernelCall", "subroutineDefinition", 
  "returnStatement", "subroutineBlock", "subroutineCall", "pragma", "timingType", 
  "timingBox", "timingTerminator", "timingIdentifier", "timingInstructionName", 
  "timingInstruction", "timingStatement", "calibration", "calibrationGrammarDeclaration", 
  "calibrationDefinition", "calibrationGrammar", "calibrationArgumentList"
};

std::vector<std::string> qasm3Parser::_literalNames = {
  "", "'OPENQASM'", "'include'", "'QCOR_EXPECT_TRUE'", "'qubit'", "'qreg'", 
  "'bit'", "'creg'", "'int'", "'uint'", "'float'", "'angle'", "'fixed'", 
  "'bool'", "'int64_t'", "'double'", "'const'", "'let'", "'||'", "'gate'", 
  "'CX'", "'U'", "'gphase'", "'measure'", "'barrier'", "'inv'", "'pow'", 
  "'ctrl'", "'@'", "'reset'", "'~'", "'!'", "'>'", "'<'", "'>='", "'<='", 
  "'=='", "'!='", "'&&'", "'|'", "'^'", "'&'", "'<<'", "'>>'", "'+'", "'*'", 
  "'/'", "'%'", "'++'", "'--'", "'sin'", "'cos'", "'tan'", "'arctan'", "'arccos'", 
  "'arcsin'", "'exp'", "'ln'", "'sqrt'", "'rotl'", "'rotr'", "'popcount'", 
  "'lengthof'", "'+='", "'-='", "'*='", "'/='", "'&='", "'|='", "'~='", 
  "'^='", "'<<='", "'>>='", "'in'", "'if'", "'else'", "'for'", "'while'", 
  "'range'", "'break'", "'continue'", "'end'", "'kernel'", "'def'", "'return'", 
  "'#pragma'", "'length'", "'stretch'", "'boxas'", "'boxto'", "'stretchinf'", 
  "'delay'", "'rotary'", "'defcalgrammar'", "'defcal'", "'\"openpulse\"'", 
  "'['", "']'", "'{'", "'}'", "'('", "')'", "':'", "';'", "'.'", "','", 
  "'='", "'->'", "'-'"
};

std::vector<std::string> qasm3Parser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE", "LPAREN", 
  "RPAREN", "COLON", "SEMICOLON", "DOT", "COMMA", "EQUALS", "ARROW", "MINUS", 
  "Constant", "Whitespace", "Newline", "Integer", "Identifier", "RealNumber", 
  "TimingLiteral", "StringLiteral", "LineComment", "BlockComment"
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
       0x3, 0x78, 0x3dc, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
       0x9, 0x68, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x7, 0x2, 0xd4, 0xa, 0x2, 
       0xc, 0x2, 0xe, 0x2, 0xd7, 0xb, 0x2, 0x3, 0x3, 0x5, 0x3, 0xda, 0xa, 
       0x3, 0x3, 0x3, 0x7, 0x3, 0xdd, 0xa, 0x3, 0xc, 0x3, 0xe, 0x3, 0xe0, 
       0xb, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 
       0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0xf0, 0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 
       0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
       0x7, 0x3, 0x7, 0x5, 0x7, 0xfc, 0xa, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 
       0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
       0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0x109, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 
       0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x111, 0xa, 0xb, 
       0x3, 0xc, 0x3, 0xc, 0x5, 0xc, 0x115, 0xa, 0xc, 0x3, 0xc, 0x3, 0xc, 
       0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 
       0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 
       0x3, 0x10, 0x3, 0x10, 0x7, 0x10, 0x128, 0xa, 0x10, 0xc, 0x10, 0xe, 
       0x10, 0x12b, 0xb, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 
       0x3, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
       0x3, 0x14, 0x3, 0x14, 0x5, 0x14, 0x139, 0xa, 0x14, 0x3, 0x14, 0x3, 
       0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x7, 0x15, 0x140, 0xa, 0x15, 
       0xc, 0x15, 0xe, 0x15, 0x143, 0xb, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 
       0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 
       0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x5, 
       0x19, 0x153, 0xa, 0x19, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 
       0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 
       0x15e, 0xa, 0x1a, 0x5, 0x1a, 0x160, 0xa, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 
       0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x5, 0x1c, 
       0x169, 0xa, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x5, 
       0x1d, 0x16f, 0xa, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x5, 0x1e, 
       0x174, 0xa, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x5, 0x1f, 0x179, 
       0xa, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x5, 0x20, 
       0x17f, 0xa, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x7, 0x21, 0x184, 
       0xa, 0x21, 0xc, 0x21, 0xe, 0x21, 0x187, 0xb, 0x21, 0x3, 0x21, 0x3, 
       0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 
       0x23, 0x7, 0x23, 0x191, 0xa, 0x23, 0xc, 0x23, 0xe, 0x23, 0x194, 0xb, 
       0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 
       0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 
       0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x5, 0x25, 0x1a6, 
       0xa, 0x25, 0x5, 0x25, 0x1a8, 0xa, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 
       0x25, 0x7, 0x25, 0x1ad, 0xa, 0x25, 0xc, 0x25, 0xe, 0x25, 0x1b0, 0xb, 
       0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x7, 0x26, 0x1b5, 0xa, 0x26, 
       0xc, 0x26, 0xe, 0x26, 0x1b8, 0xb, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 
       0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 0x1c0, 0xa, 0x27, 
       0xc, 0x27, 0xe, 0x27, 0x1c3, 0xb, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 
       0x27, 0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 0x1ca, 0xa, 0x28, 0x3, 0x28, 
       0x3, 0x28, 0x5, 0x28, 0x1ce, 0xa, 0x28, 0x3, 0x28, 0x3, 0x28, 0x5, 
       0x28, 0x1d2, 0xa, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x29, 0x3, 0x29, 
       0x3, 0x29, 0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x5, 0x2a, 
       0x1dd, 0xa, 0x2a, 0x3, 0x2a, 0x5, 0x2a, 0x1e0, 0xa, 0x2a, 0x3, 0x2a, 
       0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x7, 0x2b, 0x1e7, 0xa, 
       0x2b, 0xc, 0x2b, 0xe, 0x2b, 0x1ea, 0xb, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 
       0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 
       0x7, 0x2d, 0x1f4, 0xa, 0x2d, 0xc, 0x2d, 0xe, 0x2d, 0x1f7, 0xb, 0x2d, 
       0x3, 0x2d, 0x5, 0x2d, 0x1fa, 0xa, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 
       0x2e, 0x3, 0x2e, 0x5, 0x2e, 0x200, 0xa, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 
       0x3, 0x2f, 0x3, 0x2f, 0x5, 0x2f, 0x206, 0xa, 0x2f, 0x3, 0x30, 0x3, 
       0x30, 0x3, 0x30, 0x3, 0x30, 0x3, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 
       0x31, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x5, 0x32, 0x213, 0xa, 0x32, 
       0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x5, 0x32, 0x219, 0xa, 
       0x32, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x34, 0x3, 0x34, 0x3, 
       0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x5, 0x34, 0x225, 
       0xa, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x35, 0x3, 0x35, 0x3, 0x35, 
       0x5, 0x35, 0x22c, 0xa, 0x35, 0x3, 0x35, 0x5, 0x35, 0x22f, 0xa, 0x35, 
       0x3, 0x35, 0x3, 0x35, 0x3, 0x36, 0x3, 0x36, 0x3, 0x36, 0x3, 0x36, 
       0x3, 0x36, 0x3, 0x36, 0x3, 0x36, 0x5, 0x36, 0x23a, 0xa, 0x36, 0x3, 
       0x37, 0x3, 0x37, 0x3, 0x38, 0x3, 0x38, 0x3, 0x39, 0x3, 0x39, 0x3, 
       0x3a, 0x3, 0x3a, 0x3, 0x3a, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 
       0x3b, 0x5, 0x3b, 0x249, 0xa, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 
       0x7, 0x3b, 0x24e, 0xa, 0x3b, 0xc, 0x3b, 0xe, 0x3b, 0x251, 0xb, 0x3b, 
       0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 
       0x7, 0x3c, 0x259, 0xa, 0x3c, 0xc, 0x3c, 0xe, 0x3c, 0x25c, 0xb, 0x3c, 
       0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 
       0x7, 0x3d, 0x264, 0xa, 0x3d, 0xc, 0x3d, 0xe, 0x3d, 0x267, 0xb, 0x3d, 
       0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 
       0x7, 0x3e, 0x26f, 0xa, 0x3e, 0xc, 0x3e, 0xe, 0x3e, 0x272, 0xb, 0x3e, 
       0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 
       0x7, 0x3f, 0x27a, 0xa, 0x3f, 0xc, 0x3f, 0xe, 0x3f, 0x27d, 0xb, 0x3f, 
       0x3, 0x40, 0x3, 0x40, 0x3, 0x40, 0x5, 0x40, 0x282, 0xa, 0x40, 0x3, 
       0x40, 0x3, 0x40, 0x3, 0x40, 0x3, 0x40, 0x5, 0x40, 0x288, 0xa, 0x40, 
       0x7, 0x40, 0x28a, 0xa, 0x40, 0xc, 0x40, 0xe, 0x40, 0x28d, 0xb, 0x40, 
       0x3, 0x41, 0x3, 0x41, 0x3, 0x41, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 
       0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 
       0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 
       0x3, 0x42, 0x5, 0x42, 0x2a2, 0xa, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 
       0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x7, 0x42, 0x2ab, 
       0xa, 0x42, 0xc, 0x42, 0xe, 0x42, 0x2ae, 0xb, 0x42, 0x3, 0x43, 0x3, 
       0x43, 0x3, 0x44, 0x3, 0x44, 0x5, 0x44, 0x2b4, 0xa, 0x44, 0x3, 0x44, 
       0x3, 0x44, 0x3, 0x44, 0x3, 0x44, 0x3, 0x45, 0x3, 0x45, 0x3, 0x46, 
       0x3, 0x46, 0x3, 0x47, 0x3, 0x47, 0x3, 0x47, 0x7, 0x47, 0x2c1, 0xa, 
       0x47, 0xc, 0x47, 0xe, 0x47, 0x2c4, 0xb, 0x47, 0x3, 0x47, 0x3, 0x47, 
       0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x5, 0x48, 0x2cb, 0xa, 0x48, 0x3, 
       0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x7, 0x48, 0x2d1, 0xa, 0x48, 
       0xc, 0x48, 0xe, 0x48, 0x2d4, 0xb, 0x48, 0x3, 0x49, 0x3, 0x49, 0x3, 
       0x49, 0x3, 0x49, 0x3, 0x49, 0x5, 0x49, 0x2db, 0xa, 0x49, 0x3, 0x4a, 
       0x3, 0x4a, 0x3, 0x4a, 0x3, 0x4b, 0x3, 0x4b, 0x3, 0x4c, 0x3, 0x4c, 
       0x3, 0x4c, 0x3, 0x4c, 0x7, 0x4c, 0x2e6, 0xa, 0x4c, 0xc, 0x4c, 0xe, 
       0x4c, 0x2e9, 0xb, 0x4c, 0x3, 0x4c, 0x3, 0x4c, 0x3, 0x4c, 0x3, 0x4d, 
       0x3, 0x4d, 0x3, 0x4d, 0x3, 0x4d, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 
       0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x5, 0x4e, 0x2f8, 0xa, 0x4e, 0x3, 
       0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x7, 0x4f, 0x2fd, 0xa, 0x4f, 0xc, 0x4f, 
       0xe, 0x4f, 0x300, 0xb, 0x4f, 0x3, 0x4f, 0x5, 0x4f, 0x303, 0xa, 0x4f, 
       0x3, 0x50, 0x3, 0x50, 0x3, 0x50, 0x3, 0x50, 0x3, 0x50, 0x3, 0x50, 
       0x3, 0x50, 0x5, 0x50, 0x30c, 0xa, 0x50, 0x3, 0x51, 0x3, 0x51, 0x3, 
       0x51, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x5, 0x51, 0x315, 
       0xa, 0x51, 0x3, 0x52, 0x3, 0x52, 0x3, 0x52, 0x3, 0x53, 0x3, 0x53, 
       0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 
       0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 
       0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 
       0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 
       0x3, 0x53, 0x3, 0x53, 0x5, 0x53, 0x336, 0xa, 0x53, 0x3, 0x54, 0x3, 
       0x54, 0x3, 0x54, 0x3, 0x55, 0x3, 0x55, 0x3, 0x56, 0x3, 0x56, 0x3, 
       0x56, 0x3, 0x56, 0x5, 0x56, 0x341, 0xa, 0x56, 0x3, 0x56, 0x5, 0x56, 
       0x344, 0xa, 0x56, 0x3, 0x56, 0x5, 0x56, 0x347, 0xa, 0x56, 0x3, 0x56, 
       0x5, 0x56, 0x34a, 0xa, 0x56, 0x3, 0x56, 0x3, 0x56, 0x3, 0x57, 0x3, 
       0x57, 0x3, 0x57, 0x5, 0x57, 0x351, 0xa, 0x57, 0x3, 0x57, 0x3, 0x57, 
       0x3, 0x58, 0x3, 0x58, 0x3, 0x58, 0x3, 0x58, 0x5, 0x58, 0x359, 0xa, 
       0x58, 0x3, 0x58, 0x5, 0x58, 0x35c, 0xa, 0x58, 0x3, 0x58, 0x5, 0x58, 
       0x35f, 0xa, 0x58, 0x3, 0x58, 0x5, 0x58, 0x362, 0xa, 0x58, 0x3, 0x58, 
       0x3, 0x58, 0x3, 0x59, 0x3, 0x59, 0x3, 0x59, 0x3, 0x5a, 0x3, 0x5a, 
       0x7, 0x5a, 0x36b, 0xa, 0x5a, 0xc, 0x5a, 0xe, 0x5a, 0x36e, 0xb, 0x5a, 
       0x3, 0x5a, 0x5, 0x5a, 0x371, 0xa, 0x5a, 0x3, 0x5a, 0x3, 0x5a, 0x3, 
       0x5b, 0x3, 0x5b, 0x3, 0x5b, 0x5, 0x5b, 0x378, 0xa, 0x5b, 0x3, 0x5b, 
       0x3, 0x5b, 0x3, 0x5b, 0x3, 0x5b, 0x3, 0x5c, 0x3, 0x5c, 0x3, 0x5c, 
       0x7, 0x5c, 0x381, 0xa, 0x5c, 0xc, 0x5c, 0xe, 0x5c, 0x384, 0xb, 0x5c, 
       0x3, 0x5c, 0x3, 0x5c, 0x3, 0x5d, 0x3, 0x5d, 0x3, 0x5d, 0x5, 0x5d, 
       0x38b, 0xa, 0x5d, 0x5, 0x5d, 0x38d, 0xa, 0x5d, 0x3, 0x5e, 0x3, 0x5e, 
       0x3, 0x5e, 0x3, 0x5e, 0x3, 0x5e, 0x3, 0x5e, 0x5, 0x5e, 0x395, 0xa, 
       0x5e, 0x3, 0x5f, 0x3, 0x5f, 0x5, 0x5f, 0x399, 0xa, 0x5f, 0x3, 0x60, 
       0x3, 0x60, 0x3, 0x60, 0x3, 0x60, 0x3, 0x60, 0x5, 0x60, 0x3a0, 0xa, 
       0x60, 0x3, 0x60, 0x5, 0x60, 0x3a3, 0xa, 0x60, 0x3, 0x61, 0x3, 0x61, 
       0x3, 0x62, 0x3, 0x62, 0x3, 0x62, 0x5, 0x62, 0x3aa, 0xa, 0x62, 0x3, 
       0x62, 0x5, 0x62, 0x3ad, 0xa, 0x62, 0x3, 0x62, 0x3, 0x62, 0x3, 0x62, 
       0x3, 0x63, 0x3, 0x63, 0x3, 0x63, 0x3, 0x63, 0x5, 0x63, 0x3b6, 0xa, 
       0x63, 0x3, 0x64, 0x3, 0x64, 0x5, 0x64, 0x3ba, 0xa, 0x64, 0x3, 0x65, 
       0x3, 0x65, 0x3, 0x65, 0x3, 0x65, 0x3, 0x66, 0x3, 0x66, 0x3, 0x66, 
       0x3, 0x66, 0x5, 0x66, 0x3c4, 0xa, 0x66, 0x3, 0x66, 0x5, 0x66, 0x3c7, 
       0xa, 0x66, 0x3, 0x66, 0x3, 0x66, 0x5, 0x66, 0x3cb, 0xa, 0x66, 0x3, 
       0x66, 0x3, 0x66, 0x7, 0x66, 0x3cf, 0xa, 0x66, 0xc, 0x66, 0xe, 0x66, 
       0x3d2, 0xb, 0x66, 0x3, 0x66, 0x3, 0x66, 0x3, 0x67, 0x3, 0x67, 0x3, 
       0x68, 0x3, 0x68, 0x5, 0x68, 0x3da, 0xa, 0x68, 0x3, 0x68, 0x3, 0x3d0, 
       0xb, 0x48, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e, 0x82, 0x8e, 0x69, 
       0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 
       0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 
       0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 
       0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 
       0x5c, 0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 
       0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e, 0x80, 0x82, 0x84, 0x86, 
       0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 
       0x9e, 0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 
       0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe, 0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 
       0xca, 0xcc, 0xce, 0x2, 0x13, 0x4, 0x2, 0x72, 0x72, 0x74, 0x74, 0x3, 
       0x2, 0x6, 0x7, 0x3, 0x2, 0x8, 0x9, 0x3, 0x2, 0xa, 0xd, 0x4, 0x2, 
       0x16, 0x17, 0x73, 0x73, 0x3, 0x2, 0x20, 0x21, 0x3, 0x2, 0x22, 0x27, 
       0x4, 0x2, 0x14, 0x14, 0x28, 0x28, 0x3, 0x2, 0x2c, 0x2d, 0x4, 0x2, 
       0x2e, 0x2e, 0x6e, 0x6e, 0x3, 0x2, 0x2f, 0x31, 0x3, 0x2, 0x32, 0x33, 
       0x3, 0x2, 0x34, 0x40, 0x4, 0x2, 0x41, 0x4a, 0x6c, 0x6c, 0x3, 0x2, 
       0x51, 0x53, 0x3, 0x2, 0x5d, 0x5e, 0x4, 0x2, 0x61, 0x61, 0x76, 0x76, 
       0x2, 0x3f9, 0x2, 0xd0, 0x3, 0x2, 0x2, 0x2, 0x4, 0xd9, 0x3, 0x2, 0x2, 
       0x2, 0x6, 0xe1, 0x3, 0x2, 0x2, 0x2, 0x8, 0xe5, 0x3, 0x2, 0x2, 0x2, 
       0xa, 0xef, 0x3, 0x2, 0x2, 0x2, 0xc, 0xfb, 0x3, 0x2, 0x2, 0x2, 0xe, 
       0xfd, 0x3, 0x2, 0x2, 0x2, 0x10, 0x103, 0x3, 0x2, 0x2, 0x2, 0x12, 
       0x108, 0x3, 0x2, 0x2, 0x2, 0x14, 0x10c, 0x3, 0x2, 0x2, 0x2, 0x16, 
       0x114, 0x3, 0x2, 0x2, 0x2, 0x18, 0x118, 0x3, 0x2, 0x2, 0x2, 0x1a, 
       0x11b, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x11f, 0x3, 0x2, 0x2, 0x2, 0x1e, 
       0x129, 0x3, 0x2, 0x2, 0x2, 0x20, 0x12e, 0x3, 0x2, 0x2, 0x2, 0x22, 
       0x131, 0x3, 0x2, 0x2, 0x2, 0x24, 0x133, 0x3, 0x2, 0x2, 0x2, 0x26, 
       0x136, 0x3, 0x2, 0x2, 0x2, 0x28, 0x141, 0x3, 0x2, 0x2, 0x2, 0x2a, 
       0x146, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x148, 0x3, 0x2, 0x2, 0x2, 0x2e, 
       0x14a, 0x3, 0x2, 0x2, 0x2, 0x30, 0x152, 0x3, 0x2, 0x2, 0x2, 0x32, 
       0x15f, 0x3, 0x2, 0x2, 0x2, 0x34, 0x161, 0x3, 0x2, 0x2, 0x2, 0x36, 
       0x164, 0x3, 0x2, 0x2, 0x2, 0x38, 0x16a, 0x3, 0x2, 0x2, 0x2, 0x3a, 
       0x170, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x175, 0x3, 0x2, 0x2, 0x2, 0x3e, 
       0x17e, 0x3, 0x2, 0x2, 0x2, 0x40, 0x185, 0x3, 0x2, 0x2, 0x2, 0x42, 
       0x18a, 0x3, 0x2, 0x2, 0x2, 0x44, 0x192, 0x3, 0x2, 0x2, 0x2, 0x46, 
       0x197, 0x3, 0x2, 0x2, 0x2, 0x48, 0x1a7, 0x3, 0x2, 0x2, 0x2, 0x4a, 
       0x1b6, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x1c1, 0x3, 0x2, 0x2, 0x2, 0x4e, 
       0x1c7, 0x3, 0x2, 0x2, 0x2, 0x50, 0x1d5, 0x3, 0x2, 0x2, 0x2, 0x52, 
       0x1d9, 0x3, 0x2, 0x2, 0x2, 0x54, 0x1e3, 0x3, 0x2, 0x2, 0x2, 0x56, 
       0x1ed, 0x3, 0x2, 0x2, 0x2, 0x58, 0x1f9, 0x3, 0x2, 0x2, 0x2, 0x5a, 
       0x1ff, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x205, 0x3, 0x2, 0x2, 0x2, 0x5e, 
       0x207, 0x3, 0x2, 0x2, 0x2, 0x60, 0x20c, 0x3, 0x2, 0x2, 0x2, 0x62, 
       0x218, 0x3, 0x2, 0x2, 0x2, 0x64, 0x21a, 0x3, 0x2, 0x2, 0x2, 0x66, 
       0x224, 0x3, 0x2, 0x2, 0x2, 0x68, 0x228, 0x3, 0x2, 0x2, 0x2, 0x6a, 
       0x239, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x23b, 0x3, 0x2, 0x2, 0x2, 0x6e, 
       0x23d, 0x3, 0x2, 0x2, 0x2, 0x70, 0x23f, 0x3, 0x2, 0x2, 0x2, 0x72, 
       0x241, 0x3, 0x2, 0x2, 0x2, 0x74, 0x248, 0x3, 0x2, 0x2, 0x2, 0x76, 
       0x252, 0x3, 0x2, 0x2, 0x2, 0x78, 0x25d, 0x3, 0x2, 0x2, 0x2, 0x7a, 
       0x268, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x273, 0x3, 0x2, 0x2, 0x2, 0x7e, 
       0x281, 0x3, 0x2, 0x2, 0x2, 0x80, 0x28e, 0x3, 0x2, 0x2, 0x2, 0x82, 
       0x2a1, 0x3, 0x2, 0x2, 0x2, 0x84, 0x2af, 0x3, 0x2, 0x2, 0x2, 0x86, 
       0x2b3, 0x3, 0x2, 0x2, 0x2, 0x88, 0x2b9, 0x3, 0x2, 0x2, 0x2, 0x8a, 
       0x2bb, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x2c2, 0x3, 0x2, 0x2, 0x2, 0x8e, 
       0x2ca, 0x3, 0x2, 0x2, 0x2, 0x90, 0x2da, 0x3, 0x2, 0x2, 0x2, 0x92, 
       0x2dc, 0x3, 0x2, 0x2, 0x2, 0x94, 0x2df, 0x3, 0x2, 0x2, 0x2, 0x96, 
       0x2e7, 0x3, 0x2, 0x2, 0x2, 0x98, 0x2ed, 0x3, 0x2, 0x2, 0x2, 0x9a, 
       0x2f7, 0x3, 0x2, 0x2, 0x2, 0x9c, 0x302, 0x3, 0x2, 0x2, 0x2, 0x9e, 
       0x304, 0x3, 0x2, 0x2, 0x2, 0xa0, 0x314, 0x3, 0x2, 0x2, 0x2, 0xa2, 
       0x316, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x335, 0x3, 0x2, 0x2, 0x2, 0xa6, 
       0x337, 0x3, 0x2, 0x2, 0x2, 0xa8, 0x33a, 0x3, 0x2, 0x2, 0x2, 0xaa, 
       0x33c, 0x3, 0x2, 0x2, 0x2, 0xac, 0x34d, 0x3, 0x2, 0x2, 0x2, 0xae, 
       0x354, 0x3, 0x2, 0x2, 0x2, 0xb0, 0x365, 0x3, 0x2, 0x2, 0x2, 0xb2, 
       0x368, 0x3, 0x2, 0x2, 0x2, 0xb4, 0x374, 0x3, 0x2, 0x2, 0x2, 0xb6, 
       0x37d, 0x3, 0x2, 0x2, 0x2, 0xb8, 0x38c, 0x3, 0x2, 0x2, 0x2, 0xba, 
       0x394, 0x3, 0x2, 0x2, 0x2, 0xbc, 0x398, 0x3, 0x2, 0x2, 0x2, 0xbe, 
       0x3a2, 0x3, 0x2, 0x2, 0x2, 0xc0, 0x3a4, 0x3, 0x2, 0x2, 0x2, 0xc2, 
       0x3a6, 0x3, 0x2, 0x2, 0x2, 0xc4, 0x3b5, 0x3, 0x2, 0x2, 0x2, 0xc6, 
       0x3b9, 0x3, 0x2, 0x2, 0x2, 0xc8, 0x3bb, 0x3, 0x2, 0x2, 0x2, 0xca, 
       0x3bf, 0x3, 0x2, 0x2, 0x2, 0xcc, 0x3d5, 0x3, 0x2, 0x2, 0x2, 0xce, 
       0x3d9, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xd5, 0x5, 0x4, 0x3, 0x2, 0xd1, 
       0xd4, 0x5, 0xa, 0x6, 0x2, 0xd2, 0xd4, 0x5, 0xc, 0x7, 0x2, 0xd3, 0xd1, 
       0x3, 0x2, 0x2, 0x2, 0xd3, 0xd2, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd7, 0x3, 
       0x2, 0x2, 0x2, 0xd5, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd6, 0x3, 0x2, 
       0x2, 0x2, 0xd6, 0x3, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xd5, 0x3, 0x2, 0x2, 
       0x2, 0xd8, 0xda, 0x5, 0x6, 0x4, 0x2, 0xd9, 0xd8, 0x3, 0x2, 0x2, 0x2, 
       0xd9, 0xda, 0x3, 0x2, 0x2, 0x2, 0xda, 0xde, 0x3, 0x2, 0x2, 0x2, 0xdb, 
       0xdd, 0x5, 0x8, 0x5, 0x2, 0xdc, 0xdb, 0x3, 0x2, 0x2, 0x2, 0xdd, 0xe0, 
       0x3, 0x2, 0x2, 0x2, 0xde, 0xdc, 0x3, 0x2, 0x2, 0x2, 0xde, 0xdf, 0x3, 
       0x2, 0x2, 0x2, 0xdf, 0x5, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xde, 0x3, 0x2, 
       0x2, 0x2, 0xe1, 0xe2, 0x7, 0x3, 0x2, 0x2, 0xe2, 0xe3, 0x9, 0x2, 0x2, 
       0x2, 0xe3, 0xe4, 0x7, 0x69, 0x2, 0x2, 0xe4, 0x7, 0x3, 0x2, 0x2, 0x2, 
       0xe5, 0xe6, 0x7, 0x4, 0x2, 0x2, 0xe6, 0xe7, 0x7, 0x76, 0x2, 0x2, 
       0xe7, 0xe8, 0x7, 0x69, 0x2, 0x2, 0xe8, 0x9, 0x3, 0x2, 0x2, 0x2, 0xe9, 
       0xf0, 0x5, 0xae, 0x58, 0x2, 0xea, 0xf0, 0x5, 0xaa, 0x56, 0x2, 0xeb, 
       0xf0, 0x5, 0x50, 0x29, 0x2, 0xec, 0xf0, 0x5, 0xc6, 0x64, 0x2, 0xed, 
       0xf0, 0x5, 0x10, 0x9, 0x2, 0xee, 0xf0, 0x5, 0xb6, 0x5c, 0x2, 0xef, 
       0xe9, 0x3, 0x2, 0x2, 0x2, 0xef, 0xea, 0x3, 0x2, 0x2, 0x2, 0xef, 0xeb, 
       0x3, 0x2, 0x2, 0x2, 0xef, 0xec, 0x3, 0x2, 0x2, 0x2, 0xef, 0xed, 0x3, 
       0x2, 0x2, 0x2, 0xef, 0xee, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xb, 0x3, 0x2, 
       0x2, 0x2, 0xf1, 0xfc, 0x5, 0x72, 0x3a, 0x2, 0xf2, 0xfc, 0x5, 0x16, 
       0xc, 0x2, 0xf3, 0xfc, 0x5, 0x12, 0xa, 0x2, 0xf4, 0xfc, 0x5, 0x9e, 
       0x50, 0x2, 0xf5, 0xfc, 0x5, 0xa2, 0x52, 0x2, 0xf6, 0xfc, 0x5, 0xa6, 
       0x54, 0x2, 0xf7, 0xfc, 0x5, 0x46, 0x24, 0x2, 0xf8, 0xfc, 0x5, 0x5a, 
       0x2e, 0x2, 0xf9, 0xfc, 0x5, 0xb0, 0x59, 0x2, 0xfa, 0xfc, 0x5, 0xe, 
       0x8, 0x2, 0xfb, 0xf1, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xf2, 0x3, 0x2, 0x2, 
       0x2, 0xfb, 0xf3, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xf4, 0x3, 0x2, 0x2, 0x2, 
       0xfb, 0xf5, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xf6, 0x3, 0x2, 0x2, 0x2, 0xfb, 
       0xf7, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xf9, 
       0x3, 0x2, 0x2, 0x2, 0xfb, 0xfa, 0x3, 0x2, 0x2, 0x2, 0xfc, 0xd, 0x3, 
       0x2, 0x2, 0x2, 0xfd, 0xfe, 0x7, 0x5, 0x2, 0x2, 0xfe, 0xff, 0x7, 0x66, 
       0x2, 0x2, 0xff, 0x100, 0x5, 0x8e, 0x48, 0x2, 0x100, 0x101, 0x7, 0x67, 
       0x2, 0x2, 0x101, 0x102, 0x7, 0x69, 0x2, 0x2, 0x102, 0xf, 0x3, 0x2, 
       0x2, 0x2, 0x103, 0x104, 0x5, 0x24, 0x13, 0x2, 0x104, 0x105, 0x7, 
       0x69, 0x2, 0x2, 0x105, 0x11, 0x3, 0x2, 0x2, 0x2, 0x106, 0x109, 0x5, 
       0x3e, 0x20, 0x2, 0x107, 0x109, 0x5, 0x34, 0x1b, 0x2, 0x108, 0x106, 
       0x3, 0x2, 0x2, 0x2, 0x108, 0x107, 0x3, 0x2, 0x2, 0x2, 0x109, 0x10a, 
       0x3, 0x2, 0x2, 0x2, 0x10a, 0x10b, 0x7, 0x69, 0x2, 0x2, 0x10b, 0x13, 
       0x3, 0x2, 0x2, 0x2, 0x10c, 0x10d, 0x5, 0x48, 0x25, 0x2, 0x10d, 0x110, 
       0x5, 0x94, 0x4b, 0x2, 0x10e, 0x111, 0x5, 0x74, 0x3b, 0x2, 0x10f, 
       0x111, 0x5, 0x48, 0x25, 0x2, 0x110, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x110, 
       0x10f, 0x3, 0x2, 0x2, 0x2, 0x111, 0x15, 0x3, 0x2, 0x2, 0x2, 0x112, 
       0x115, 0x5, 0x14, 0xb, 0x2, 0x113, 0x115, 0x5, 0x62, 0x32, 0x2, 0x114, 
       0x112, 0x3, 0x2, 0x2, 0x2, 0x114, 0x113, 0x3, 0x2, 0x2, 0x2, 0x115, 
       0x116, 0x3, 0x2, 0x2, 0x2, 0x116, 0x117, 0x7, 0x69, 0x2, 0x2, 0x117, 
       0x17, 0x3, 0x2, 0x2, 0x2, 0x118, 0x119, 0x7, 0x6d, 0x2, 0x2, 0x119, 
       0x11a, 0x5, 0x32, 0x1a, 0x2, 0x11a, 0x19, 0x3, 0x2, 0x2, 0x2, 0x11b, 
       0x11c, 0x7, 0x62, 0x2, 0x2, 0x11c, 0x11d, 0x5, 0x74, 0x3b, 0x2, 0x11d, 
       0x11e, 0x7, 0x63, 0x2, 0x2, 0x11e, 0x1b, 0x3, 0x2, 0x2, 0x2, 0x11f, 
       0x120, 0x7, 0x62, 0x2, 0x2, 0x120, 0x121, 0x5, 0x74, 0x3b, 0x2, 0x121, 
       0x122, 0x7, 0x6b, 0x2, 0x2, 0x122, 0x123, 0x5, 0x74, 0x3b, 0x2, 0x123, 
       0x124, 0x7, 0x63, 0x2, 0x2, 0x124, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x125, 
       0x126, 0x7, 0x73, 0x2, 0x2, 0x126, 0x128, 0x7, 0x6b, 0x2, 0x2, 0x127, 
       0x125, 0x3, 0x2, 0x2, 0x2, 0x128, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x129, 
       0x127, 0x3, 0x2, 0x2, 0x2, 0x129, 0x12a, 0x3, 0x2, 0x2, 0x2, 0x12a, 
       0x12c, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x129, 0x3, 0x2, 0x2, 0x2, 0x12c, 
       0x12d, 0x7, 0x73, 0x2, 0x2, 0x12d, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x12e, 
       0x12f, 0x7, 0x68, 0x2, 0x2, 0x12f, 0x130, 0x7, 0x73, 0x2, 0x2, 0x130, 
       0x21, 0x3, 0x2, 0x2, 0x2, 0x131, 0x132, 0x9, 0x3, 0x2, 0x2, 0x132, 
       0x23, 0x3, 0x2, 0x2, 0x2, 0x133, 0x134, 0x5, 0x22, 0x12, 0x2, 0x134, 
       0x135, 0x5, 0x4a, 0x26, 0x2, 0x135, 0x25, 0x3, 0x2, 0x2, 0x2, 0x136, 
       0x138, 0x5, 0x22, 0x12, 0x2, 0x137, 0x139, 0x5, 0x1a, 0xe, 0x2, 0x138, 
       0x137, 0x3, 0x2, 0x2, 0x2, 0x138, 0x139, 0x3, 0x2, 0x2, 0x2, 0x139, 
       0x13a, 0x3, 0x2, 0x2, 0x2, 0x13a, 0x13b, 0x5, 0x20, 0x11, 0x2, 0x13b, 
       0x27, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x13d, 0x5, 0x26, 0x14, 0x2, 0x13d, 
       0x13e, 0x7, 0x6b, 0x2, 0x2, 0x13e, 0x140, 0x3, 0x2, 0x2, 0x2, 0x13f, 
       0x13c, 0x3, 0x2, 0x2, 0x2, 0x140, 0x143, 0x3, 0x2, 0x2, 0x2, 0x141, 
       0x13f, 0x3, 0x2, 0x2, 0x2, 0x141, 0x142, 0x3, 0x2, 0x2, 0x2, 0x142, 
       0x144, 0x3, 0x2, 0x2, 0x2, 0x143, 0x141, 0x3, 0x2, 0x2, 0x2, 0x144, 
       0x145, 0x5, 0x26, 0x14, 0x2, 0x145, 0x29, 0x3, 0x2, 0x2, 0x2, 0x146, 
       0x147, 0x9, 0x4, 0x2, 0x2, 0x147, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x148, 
       0x149, 0x9, 0x5, 0x2, 0x2, 0x149, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x14a, 
       0x14b, 0x7, 0xe, 0x2, 0x2, 0x14b, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x14c, 
       0x153, 0x7, 0xf, 0x2, 0x2, 0x14d, 0x153, 0x5, 0xb8, 0x5d, 0x2, 0x14e, 
       0x153, 0x7, 0xa, 0x2, 0x2, 0x14f, 0x153, 0x7, 0x10, 0x2, 0x2, 0x150, 
       0x153, 0x7, 0xc, 0x2, 0x2, 0x151, 0x153, 0x7, 0x11, 0x2, 0x2, 0x152, 
       0x14c, 0x3, 0x2, 0x2, 0x2, 0x152, 0x14d, 0x3, 0x2, 0x2, 0x2, 0x152, 
       0x14e, 0x3, 0x2, 0x2, 0x2, 0x152, 0x14f, 0x3, 0x2, 0x2, 0x2, 0x152, 
       0x150, 0x3, 0x2, 0x2, 0x2, 0x152, 0x151, 0x3, 0x2, 0x2, 0x2, 0x153, 
       0x31, 0x3, 0x2, 0x2, 0x2, 0x154, 0x155, 0x5, 0x2c, 0x17, 0x2, 0x155, 
       0x156, 0x5, 0x1a, 0xe, 0x2, 0x156, 0x160, 0x3, 0x2, 0x2, 0x2, 0x157, 
       0x158, 0x5, 0x2e, 0x18, 0x2, 0x158, 0x159, 0x5, 0x1c, 0xf, 0x2, 0x159, 
       0x160, 0x3, 0x2, 0x2, 0x2, 0x15a, 0x160, 0x5, 0x30, 0x19, 0x2, 0x15b, 
       0x15d, 0x5, 0x2a, 0x16, 0x2, 0x15c, 0x15e, 0x5, 0x1a, 0xe, 0x2, 0x15d, 
       0x15c, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15e, 0x3, 0x2, 0x2, 0x2, 0x15e, 
       0x160, 0x3, 0x2, 0x2, 0x2, 0x15f, 0x154, 0x3, 0x2, 0x2, 0x2, 0x15f, 
       0x157, 0x3, 0x2, 0x2, 0x2, 0x15f, 0x15a, 0x3, 0x2, 0x2, 0x2, 0x15f, 
       0x15b, 0x3, 0x2, 0x2, 0x2, 0x160, 0x33, 0x3, 0x2, 0x2, 0x2, 0x161, 
       0x162, 0x7, 0x12, 0x2, 0x2, 0x162, 0x163, 0x5, 0x96, 0x4c, 0x2, 0x163, 
       0x35, 0x3, 0x2, 0x2, 0x2, 0x164, 0x165, 0x5, 0x2c, 0x17, 0x2, 0x165, 
       0x168, 0x5, 0x1a, 0xe, 0x2, 0x166, 0x169, 0x5, 0x1e, 0x10, 0x2, 0x167, 
       0x169, 0x5, 0x96, 0x4c, 0x2, 0x168, 0x166, 0x3, 0x2, 0x2, 0x2, 0x168, 
       0x167, 0x3, 0x2, 0x2, 0x2, 0x169, 0x37, 0x3, 0x2, 0x2, 0x2, 0x16a, 
       0x16b, 0x5, 0x2e, 0x18, 0x2, 0x16b, 0x16e, 0x5, 0x1c, 0xf, 0x2, 0x16c, 
       0x16f, 0x5, 0x1e, 0x10, 0x2, 0x16d, 0x16f, 0x5, 0x96, 0x4c, 0x2, 
       0x16e, 0x16c, 0x3, 0x2, 0x2, 0x2, 0x16e, 0x16d, 0x3, 0x2, 0x2, 0x2, 
       0x16f, 0x39, 0x3, 0x2, 0x2, 0x2, 0x170, 0x173, 0x5, 0x30, 0x19, 0x2, 
       0x171, 0x174, 0x5, 0x1e, 0x10, 0x2, 0x172, 0x174, 0x5, 0x96, 0x4c, 
       0x2, 0x173, 0x171, 0x3, 0x2, 0x2, 0x2, 0x173, 0x172, 0x3, 0x2, 0x2, 
       0x2, 0x174, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x175, 0x178, 0x5, 0x2a, 0x16, 
       0x2, 0x176, 0x179, 0x5, 0x4a, 0x26, 0x2, 0x177, 0x179, 0x5, 0x4c, 
       0x27, 0x2, 0x178, 0x176, 0x3, 0x2, 0x2, 0x2, 0x178, 0x177, 0x3, 0x2, 
       0x2, 0x2, 0x179, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x17a, 0x17f, 0x5, 0x36, 
       0x1c, 0x2, 0x17b, 0x17f, 0x5, 0x38, 0x1d, 0x2, 0x17c, 0x17f, 0x5, 
       0x3a, 0x1e, 0x2, 0x17d, 0x17f, 0x5, 0x3c, 0x1f, 0x2, 0x17e, 0x17a, 
       0x3, 0x2, 0x2, 0x2, 0x17e, 0x17b, 0x3, 0x2, 0x2, 0x2, 0x17e, 0x17c, 
       0x3, 0x2, 0x2, 0x2, 0x17e, 0x17d, 0x3, 0x2, 0x2, 0x2, 0x17f, 0x3f, 
       0x3, 0x2, 0x2, 0x2, 0x180, 0x181, 0x5, 0x32, 0x1a, 0x2, 0x181, 0x182, 
       0x7, 0x6b, 0x2, 0x2, 0x182, 0x184, 0x3, 0x2, 0x2, 0x2, 0x183, 0x180, 
       0x3, 0x2, 0x2, 0x2, 0x184, 0x187, 0x3, 0x2, 0x2, 0x2, 0x185, 0x183, 
       0x3, 0x2, 0x2, 0x2, 0x185, 0x186, 0x3, 0x2, 0x2, 0x2, 0x186, 0x188, 
       0x3, 0x2, 0x2, 0x2, 0x187, 0x185, 0x3, 0x2, 0x2, 0x2, 0x188, 0x189, 
       0x5, 0x32, 0x1a, 0x2, 0x189, 0x41, 0x3, 0x2, 0x2, 0x2, 0x18a, 0x18b, 
       0x5, 0x32, 0x1a, 0x2, 0x18b, 0x18c, 0x5, 0x20, 0x11, 0x2, 0x18c, 
       0x43, 0x3, 0x2, 0x2, 0x2, 0x18d, 0x18e, 0x5, 0x42, 0x22, 0x2, 0x18e, 
       0x18f, 0x7, 0x6b, 0x2, 0x2, 0x18f, 0x191, 0x3, 0x2, 0x2, 0x2, 0x190, 
       0x18d, 0x3, 0x2, 0x2, 0x2, 0x191, 0x194, 0x3, 0x2, 0x2, 0x2, 0x192, 
       0x190, 0x3, 0x2, 0x2, 0x2, 0x192, 0x193, 0x3, 0x2, 0x2, 0x2, 0x193, 
       0x195, 0x3, 0x2, 0x2, 0x2, 0x194, 0x192, 0x3, 0x2, 0x2, 0x2, 0x195, 
       0x196, 0x5, 0x42, 0x22, 0x2, 0x196, 0x45, 0x3, 0x2, 0x2, 0x2, 0x197, 
       0x198, 0x7, 0x13, 0x2, 0x2, 0x198, 0x199, 0x7, 0x73, 0x2, 0x2, 0x199, 
       0x19a, 0x7, 0x6c, 0x2, 0x2, 0x19a, 0x19b, 0x5, 0x48, 0x25, 0x2, 0x19b, 
       0x19c, 0x7, 0x69, 0x2, 0x2, 0x19c, 0x47, 0x3, 0x2, 0x2, 0x2, 0x19d, 
       0x19e, 0x8, 0x25, 0x1, 0x2, 0x19e, 0x19f, 0x7, 0x73, 0x2, 0x2, 0x19f, 
       0x1a8, 0x5, 0x4e, 0x28, 0x2, 0x1a0, 0x1a5, 0x7, 0x73, 0x2, 0x2, 0x1a1, 
       0x1a2, 0x7, 0x62, 0x2, 0x2, 0x1a2, 0x1a3, 0x5, 0x8c, 0x47, 0x2, 0x1a3, 
       0x1a4, 0x7, 0x63, 0x2, 0x2, 0x1a4, 0x1a6, 0x3, 0x2, 0x2, 0x2, 0x1a5, 
       0x1a1, 0x3, 0x2, 0x2, 0x2, 0x1a5, 0x1a6, 0x3, 0x2, 0x2, 0x2, 0x1a6, 
       0x1a8, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x19d, 0x3, 0x2, 0x2, 0x2, 0x1a7, 
       0x1a0, 0x3, 0x2, 0x2, 0x2, 0x1a8, 0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1a9, 
       0x1aa, 0xc, 0x3, 0x2, 0x2, 0x1aa, 0x1ab, 0x7, 0x14, 0x2, 0x2, 0x1ab, 
       0x1ad, 0x5, 0x48, 0x25, 0x4, 0x1ac, 0x1a9, 0x3, 0x2, 0x2, 0x2, 0x1ad, 
       0x1b0, 0x3, 0x2, 0x2, 0x2, 0x1ae, 0x1ac, 0x3, 0x2, 0x2, 0x2, 0x1ae, 
       0x1af, 0x3, 0x2, 0x2, 0x2, 0x1af, 0x49, 0x3, 0x2, 0x2, 0x2, 0x1b0, 
       0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x1b2, 0x5, 0x48, 0x25, 0x2, 0x1b2, 
       0x1b3, 0x7, 0x6b, 0x2, 0x2, 0x1b3, 0x1b5, 0x3, 0x2, 0x2, 0x2, 0x1b4, 
       0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b5, 0x1b8, 0x3, 0x2, 0x2, 0x2, 0x1b6, 
       0x1b4, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b7, 0x3, 0x2, 0x2, 0x2, 0x1b7, 
       0x1b9, 0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1b6, 0x3, 0x2, 0x2, 0x2, 0x1b9, 
       0x1ba, 0x5, 0x48, 0x25, 0x2, 0x1ba, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x1bb, 
       0x1bc, 0x5, 0x48, 0x25, 0x2, 0x1bc, 0x1bd, 0x5, 0x92, 0x4a, 0x2, 
       0x1bd, 0x1be, 0x7, 0x6b, 0x2, 0x2, 0x1be, 0x1c0, 0x3, 0x2, 0x2, 0x2, 
       0x1bf, 0x1bb, 0x3, 0x2, 0x2, 0x2, 0x1c0, 0x1c3, 0x3, 0x2, 0x2, 0x2, 
       0x1c1, 0x1bf, 0x3, 0x2, 0x2, 0x2, 0x1c1, 0x1c2, 0x3, 0x2, 0x2, 0x2, 
       0x1c2, 0x1c4, 0x3, 0x2, 0x2, 0x2, 0x1c3, 0x1c1, 0x3, 0x2, 0x2, 0x2, 
       0x1c4, 0x1c5, 0x5, 0x48, 0x25, 0x2, 0x1c5, 0x1c6, 0x5, 0x92, 0x4a, 
       0x2, 0x1c6, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1c9, 0x7, 0x62, 0x2, 
       0x2, 0x1c8, 0x1ca, 0x5, 0x74, 0x3b, 0x2, 0x1c9, 0x1c8, 0x3, 0x2, 
       0x2, 0x2, 0x1c9, 0x1ca, 0x3, 0x2, 0x2, 0x2, 0x1ca, 0x1cb, 0x3, 0x2, 
       0x2, 0x2, 0x1cb, 0x1cd, 0x7, 0x68, 0x2, 0x2, 0x1cc, 0x1ce, 0x5, 0x74, 
       0x3b, 0x2, 0x1cd, 0x1cc, 0x3, 0x2, 0x2, 0x2, 0x1cd, 0x1ce, 0x3, 0x2, 
       0x2, 0x2, 0x1ce, 0x1d1, 0x3, 0x2, 0x2, 0x2, 0x1cf, 0x1d0, 0x7, 0x68, 
       0x2, 0x2, 0x1d0, 0x1d2, 0x5, 0x74, 0x3b, 0x2, 0x1d1, 0x1cf, 0x3, 
       0x2, 0x2, 0x2, 0x1d1, 0x1d2, 0x3, 0x2, 0x2, 0x2, 0x1d2, 0x1d3, 0x3, 
       0x2, 0x2, 0x2, 0x1d3, 0x1d4, 0x7, 0x63, 0x2, 0x2, 0x1d4, 0x4f, 0x3, 
       0x2, 0x2, 0x2, 0x1d5, 0x1d6, 0x7, 0x15, 0x2, 0x2, 0x1d6, 0x1d7, 0x5, 
       0x52, 0x2a, 0x2, 0x1d7, 0x1d8, 0x5, 0x54, 0x2b, 0x2, 0x1d8, 0x51, 
       0x3, 0x2, 0x2, 0x2, 0x1d9, 0x1df, 0x9, 0x6, 0x2, 0x2, 0x1da, 0x1dc, 
       0x7, 0x66, 0x2, 0x2, 0x1db, 0x1dd, 0x5, 0x1e, 0x10, 0x2, 0x1dc, 0x1db, 
       0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1dd, 0x3, 0x2, 0x2, 0x2, 0x1dd, 0x1de, 
       0x3, 0x2, 0x2, 0x2, 0x1de, 0x1e0, 0x7, 0x67, 0x2, 0x2, 0x1df, 0x1da, 
       0x3, 0x2, 0x2, 0x2, 0x1df, 0x1e0, 0x3, 0x2, 0x2, 0x2, 0x1e0, 0x1e1, 
       0x3, 0x2, 0x2, 0x2, 0x1e1, 0x1e2, 0x5, 0x1e, 0x10, 0x2, 0x1e2, 0x53, 
       0x3, 0x2, 0x2, 0x2, 0x1e3, 0x1e8, 0x7, 0x64, 0x2, 0x2, 0x1e4, 0x1e7, 
       0x5, 0x5a, 0x2e, 0x2, 0x1e5, 0x1e7, 0x5, 0x56, 0x2c, 0x2, 0x1e6, 
       0x1e4, 0x3, 0x2, 0x2, 0x2, 0x1e6, 0x1e5, 0x3, 0x2, 0x2, 0x2, 0x1e7, 
       0x1ea, 0x3, 0x2, 0x2, 0x2, 0x1e8, 0x1e6, 0x3, 0x2, 0x2, 0x2, 0x1e8, 
       0x1e9, 0x3, 0x2, 0x2, 0x2, 0x1e9, 0x1eb, 0x3, 0x2, 0x2, 0x2, 0x1ea, 
       0x1e8, 0x3, 0x2, 0x2, 0x2, 0x1eb, 0x1ec, 0x7, 0x65, 0x2, 0x2, 0x1ec, 
       0x55, 0x3, 0x2, 0x2, 0x2, 0x1ed, 0x1ee, 0x5, 0xa0, 0x51, 0x2, 0x1ee, 
       0x1ef, 0x5, 0x58, 0x2d, 0x2, 0x1ef, 0x57, 0x3, 0x2, 0x2, 0x2, 0x1f0, 
       0x1fa, 0x5, 0x5a, 0x2e, 0x2, 0x1f1, 0x1f5, 0x7, 0x64, 0x2, 0x2, 0x1f2, 
       0x1f4, 0x5, 0x5a, 0x2e, 0x2, 0x1f3, 0x1f2, 0x3, 0x2, 0x2, 0x2, 0x1f4, 
       0x1f7, 0x3, 0x2, 0x2, 0x2, 0x1f5, 0x1f3, 0x3, 0x2, 0x2, 0x2, 0x1f5, 
       0x1f6, 0x3, 0x2, 0x2, 0x2, 0x1f6, 0x1f8, 0x3, 0x2, 0x2, 0x2, 0x1f7, 
       0x1f5, 0x3, 0x2, 0x2, 0x2, 0x1f8, 0x1fa, 0x7, 0x65, 0x2, 0x2, 0x1f9, 
       0x1f0, 0x3, 0x2, 0x2, 0x2, 0x1f9, 0x1f1, 0x3, 0x2, 0x2, 0x2, 0x1fa, 
       0x59, 0x3, 0x2, 0x2, 0x2, 0x1fb, 0x1fc, 0x5, 0x5c, 0x2f, 0x2, 0x1fc, 
       0x1fd, 0x7, 0x69, 0x2, 0x2, 0x1fd, 0x200, 0x3, 0x2, 0x2, 0x2, 0x1fe, 
       0x200, 0x5, 0xc4, 0x63, 0x2, 0x1ff, 0x1fb, 0x3, 0x2, 0x2, 0x2, 0x1ff, 
       0x1fe, 0x3, 0x2, 0x2, 0x2, 0x200, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x201, 
       0x206, 0x5, 0x68, 0x35, 0x2, 0x202, 0x206, 0x5, 0x5e, 0x30, 0x2, 
       0x203, 0x206, 0x5, 0x60, 0x31, 0x2, 0x204, 0x206, 0x5, 0x64, 0x33, 
       0x2, 0x205, 0x201, 0x3, 0x2, 0x2, 0x2, 0x205, 0x202, 0x3, 0x2, 0x2, 
       0x2, 0x205, 0x203, 0x3, 0x2, 0x2, 0x2, 0x205, 0x204, 0x3, 0x2, 0x2, 
       0x2, 0x206, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x207, 0x208, 0x7, 0x18, 0x2, 
       0x2, 0x208, 0x209, 0x7, 0x66, 0x2, 0x2, 0x209, 0x20a, 0x7, 0x73, 
       0x2, 0x2, 0x20a, 0x20b, 0x7, 0x67, 0x2, 0x2, 0x20b, 0x5f, 0x3, 0x2, 
       0x2, 0x2, 0x20c, 0x20d, 0x7, 0x19, 0x2, 0x2, 0x20d, 0x20e, 0x5, 0x4a, 
       0x26, 0x2, 0x20e, 0x61, 0x3, 0x2, 0x2, 0x2, 0x20f, 0x212, 0x5, 0x60, 
       0x31, 0x2, 0x210, 0x211, 0x7, 0x6d, 0x2, 0x2, 0x211, 0x213, 0x5, 
       0x4a, 0x26, 0x2, 0x212, 0x210, 0x3, 0x2, 0x2, 0x2, 0x212, 0x213, 
       0x3, 0x2, 0x2, 0x2, 0x213, 0x219, 0x3, 0x2, 0x2, 0x2, 0x214, 0x215, 
       0x5, 0x4a, 0x26, 0x2, 0x215, 0x216, 0x7, 0x6c, 0x2, 0x2, 0x216, 0x217, 
       0x5, 0x60, 0x31, 0x2, 0x217, 0x219, 0x3, 0x2, 0x2, 0x2, 0x218, 0x20f, 
       0x3, 0x2, 0x2, 0x2, 0x218, 0x214, 0x3, 0x2, 0x2, 0x2, 0x219, 0x63, 
       0x3, 0x2, 0x2, 0x2, 0x21a, 0x21b, 0x7, 0x1a, 0x2, 0x2, 0x21b, 0x21c, 
       0x5, 0x4a, 0x26, 0x2, 0x21c, 0x65, 0x3, 0x2, 0x2, 0x2, 0x21d, 0x225, 
       0x7, 0x1b, 0x2, 0x2, 0x21e, 0x21f, 0x7, 0x1c, 0x2, 0x2, 0x21f, 0x220, 
       0x7, 0x66, 0x2, 0x2, 0x220, 0x221, 0x5, 0x74, 0x3b, 0x2, 0x221, 0x222, 
       0x7, 0x67, 0x2, 0x2, 0x222, 0x225, 0x3, 0x2, 0x2, 0x2, 0x223, 0x225, 
       0x7, 0x1d, 0x2, 0x2, 0x224, 0x21d, 0x3, 0x2, 0x2, 0x2, 0x224, 0x21e, 
       0x3, 0x2, 0x2, 0x2, 0x224, 0x223, 0x3, 0x2, 0x2, 0x2, 0x225, 0x226, 
       0x3, 0x2, 0x2, 0x2, 0x226, 0x227, 0x7, 0x1e, 0x2, 0x2, 0x227, 0x67, 
       0x3, 0x2, 0x2, 0x2, 0x228, 0x22e, 0x5, 0x6a, 0x36, 0x2, 0x229, 0x22b, 
       0x7, 0x66, 0x2, 0x2, 0x22a, 0x22c, 0x5, 0x8c, 0x47, 0x2, 0x22b, 0x22a, 
       0x3, 0x2, 0x2, 0x2, 0x22b, 0x22c, 0x3, 0x2, 0x2, 0x2, 0x22c, 0x22d, 
       0x3, 0x2, 0x2, 0x2, 0x22d, 0x22f, 0x7, 0x67, 0x2, 0x2, 0x22e, 0x229, 
       0x3, 0x2, 0x2, 0x2, 0x22e, 0x22f, 0x3, 0x2, 0x2, 0x2, 0x22f, 0x230, 
       0x3, 0x2, 0x2, 0x2, 0x230, 0x231, 0x5, 0x4a, 0x26, 0x2, 0x231, 0x69, 
       0x3, 0x2, 0x2, 0x2, 0x232, 0x23a, 0x7, 0x16, 0x2, 0x2, 0x233, 0x23a, 
       0x7, 0x17, 0x2, 0x2, 0x234, 0x23a, 0x7, 0x1f, 0x2, 0x2, 0x235, 0x23a, 
       0x7, 0x73, 0x2, 0x2, 0x236, 0x237, 0x5, 0x66, 0x34, 0x2, 0x237, 0x238, 
       0x5, 0x6a, 0x36, 0x2, 0x238, 0x23a, 0x3, 0x2, 0x2, 0x2, 0x239, 0x232, 
       0x3, 0x2, 0x2, 0x2, 0x239, 0x233, 0x3, 0x2, 0x2, 0x2, 0x239, 0x234, 
       0x3, 0x2, 0x2, 0x2, 0x239, 0x235, 0x3, 0x2, 0x2, 0x2, 0x239, 0x236, 
       0x3, 0x2, 0x2, 0x2, 0x23a, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x23b, 0x23c, 
       0x9, 0x7, 0x2, 0x2, 0x23c, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x23d, 0x23e, 
       0x9, 0x8, 0x2, 0x2, 0x23e, 0x6f, 0x3, 0x2, 0x2, 0x2, 0x23f, 0x240, 
       0x9, 0x9, 0x2, 0x2, 0x240, 0x71, 0x3, 0x2, 0x2, 0x2, 0x241, 0x242, 
       0x5, 0x74, 0x3b, 0x2, 0x242, 0x243, 0x7, 0x69, 0x2, 0x2, 0x243, 0x73, 
       0x3, 0x2, 0x2, 0x2, 0x244, 0x245, 0x8, 0x3b, 0x1, 0x2, 0x245, 0x249, 
       0x5, 0x82, 0x42, 0x2, 0x246, 0x249, 0x5, 0x80, 0x41, 0x2, 0x247, 
       0x249, 0x5, 0x76, 0x3c, 0x2, 0x248, 0x244, 0x3, 0x2, 0x2, 0x2, 0x248, 
       0x246, 0x3, 0x2, 0x2, 0x2, 0x248, 0x247, 0x3, 0x2, 0x2, 0x2, 0x249, 
       0x24f, 0x3, 0x2, 0x2, 0x2, 0x24a, 0x24b, 0xc, 0x3, 0x2, 0x2, 0x24b, 
       0x24c, 0x7, 0x29, 0x2, 0x2, 0x24c, 0x24e, 0x5, 0x76, 0x3c, 0x2, 0x24d, 
       0x24a, 0x3, 0x2, 0x2, 0x2, 0x24e, 0x251, 0x3, 0x2, 0x2, 0x2, 0x24f, 
       0x24d, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x250, 0x3, 0x2, 0x2, 0x2, 0x250, 
       0x75, 0x3, 0x2, 0x2, 0x2, 0x251, 0x24f, 0x3, 0x2, 0x2, 0x2, 0x252, 
       0x253, 0x8, 0x3c, 0x1, 0x2, 0x253, 0x254, 0x5, 0x78, 0x3d, 0x2, 0x254, 
       0x25a, 0x3, 0x2, 0x2, 0x2, 0x255, 0x256, 0xc, 0x3, 0x2, 0x2, 0x256, 
       0x257, 0x7, 0x2a, 0x2, 0x2, 0x257, 0x259, 0x5, 0x78, 0x3d, 0x2, 0x258, 
       0x255, 0x3, 0x2, 0x2, 0x2, 0x259, 0x25c, 0x3, 0x2, 0x2, 0x2, 0x25a, 
       0x258, 0x3, 0x2, 0x2, 0x2, 0x25a, 0x25b, 0x3, 0x2, 0x2, 0x2, 0x25b, 
       0x77, 0x3, 0x2, 0x2, 0x2, 0x25c, 0x25a, 0x3, 0x2, 0x2, 0x2, 0x25d, 
       0x25e, 0x8, 0x3d, 0x1, 0x2, 0x25e, 0x25f, 0x5, 0x7a, 0x3e, 0x2, 0x25f, 
       0x265, 0x3, 0x2, 0x2, 0x2, 0x260, 0x261, 0xc, 0x3, 0x2, 0x2, 0x261, 
       0x262, 0x7, 0x2b, 0x2, 0x2, 0x262, 0x264, 0x5, 0x7a, 0x3e, 0x2, 0x263, 
       0x260, 0x3, 0x2, 0x2, 0x2, 0x264, 0x267, 0x3, 0x2, 0x2, 0x2, 0x265, 
       0x263, 0x3, 0x2, 0x2, 0x2, 0x265, 0x266, 0x3, 0x2, 0x2, 0x2, 0x266, 
       0x79, 0x3, 0x2, 0x2, 0x2, 0x267, 0x265, 0x3, 0x2, 0x2, 0x2, 0x268, 
       0x269, 0x8, 0x3e, 0x1, 0x2, 0x269, 0x26a, 0x5, 0x7c, 0x3f, 0x2, 0x26a, 
       0x270, 0x3, 0x2, 0x2, 0x2, 0x26b, 0x26c, 0xc, 0x3, 0x2, 0x2, 0x26c, 
       0x26d, 0x9, 0xa, 0x2, 0x2, 0x26d, 0x26f, 0x5, 0x7c, 0x3f, 0x2, 0x26e, 
       0x26b, 0x3, 0x2, 0x2, 0x2, 0x26f, 0x272, 0x3, 0x2, 0x2, 0x2, 0x270, 
       0x26e, 0x3, 0x2, 0x2, 0x2, 0x270, 0x271, 0x3, 0x2, 0x2, 0x2, 0x271, 
       0x7b, 0x3, 0x2, 0x2, 0x2, 0x272, 0x270, 0x3, 0x2, 0x2, 0x2, 0x273, 
       0x274, 0x8, 0x3f, 0x1, 0x2, 0x274, 0x275, 0x5, 0x7e, 0x40, 0x2, 0x275, 
       0x27b, 0x3, 0x2, 0x2, 0x2, 0x276, 0x277, 0xc, 0x3, 0x2, 0x2, 0x277, 
       0x278, 0x9, 0xb, 0x2, 0x2, 0x278, 0x27a, 0x5, 0x7e, 0x40, 0x2, 0x279, 
       0x276, 0x3, 0x2, 0x2, 0x2, 0x27a, 0x27d, 0x3, 0x2, 0x2, 0x2, 0x27b, 
       0x279, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x27c, 0x3, 0x2, 0x2, 0x2, 0x27c, 
       0x7d, 0x3, 0x2, 0x2, 0x2, 0x27d, 0x27b, 0x3, 0x2, 0x2, 0x2, 0x27e, 
       0x27f, 0x8, 0x40, 0x1, 0x2, 0x27f, 0x282, 0x5, 0x82, 0x42, 0x2, 0x280, 
       0x282, 0x5, 0x80, 0x41, 0x2, 0x281, 0x27e, 0x3, 0x2, 0x2, 0x2, 0x281, 
       0x280, 0x3, 0x2, 0x2, 0x2, 0x282, 0x28b, 0x3, 0x2, 0x2, 0x2, 0x283, 
       0x284, 0xc, 0x3, 0x2, 0x2, 0x284, 0x287, 0x9, 0xc, 0x2, 0x2, 0x285, 
       0x288, 0x5, 0x82, 0x42, 0x2, 0x286, 0x288, 0x5, 0x80, 0x41, 0x2, 
       0x287, 0x285, 0x3, 0x2, 0x2, 0x2, 0x287, 0x286, 0x3, 0x2, 0x2, 0x2, 
       0x288, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x289, 0x283, 0x3, 0x2, 0x2, 0x2, 
       0x28a, 0x28d, 0x3, 0x2, 0x2, 0x2, 0x28b, 0x289, 0x3, 0x2, 0x2, 0x2, 
       0x28b, 0x28c, 0x3, 0x2, 0x2, 0x2, 0x28c, 0x7f, 0x3, 0x2, 0x2, 0x2, 
       0x28d, 0x28b, 0x3, 0x2, 0x2, 0x2, 0x28e, 0x28f, 0x5, 0x6c, 0x37, 
       0x2, 0x28f, 0x290, 0x5, 0x82, 0x42, 0x2, 0x290, 0x81, 0x3, 0x2, 0x2, 
       0x2, 0x291, 0x292, 0x8, 0x42, 0x1, 0x2, 0x292, 0x2a2, 0x7, 0x6f, 
       0x2, 0x2, 0x293, 0x2a2, 0x7, 0x72, 0x2, 0x2, 0x294, 0x2a2, 0x7, 0x74, 
       0x2, 0x2, 0x295, 0x2a2, 0x7, 0x73, 0x2, 0x2, 0x296, 0x2a2, 0x7, 0x76, 
       0x2, 0x2, 0x297, 0x2a2, 0x5, 0x86, 0x44, 0x2, 0x298, 0x2a2, 0x5, 
       0xac, 0x57, 0x2, 0x299, 0x2a2, 0x5, 0xb4, 0x5b, 0x2, 0x29a, 0x2a2, 
       0x5, 0xbc, 0x5f, 0x2, 0x29b, 0x29c, 0x7, 0x6e, 0x2, 0x2, 0x29c, 0x2a2, 
       0x5, 0x82, 0x42, 0x6, 0x29d, 0x29e, 0x7, 0x66, 0x2, 0x2, 0x29e, 0x29f, 
       0x5, 0x74, 0x3b, 0x2, 0x29f, 0x2a0, 0x7, 0x67, 0x2, 0x2, 0x2a0, 0x2a2, 
       0x3, 0x2, 0x2, 0x2, 0x2a1, 0x291, 0x3, 0x2, 0x2, 0x2, 0x2a1, 0x293, 
       0x3, 0x2, 0x2, 0x2, 0x2a1, 0x294, 0x3, 0x2, 0x2, 0x2, 0x2a1, 0x295, 
       0x3, 0x2, 0x2, 0x2, 0x2a1, 0x296, 0x3, 0x2, 0x2, 0x2, 0x2a1, 0x297, 
       0x3, 0x2, 0x2, 0x2, 0x2a1, 0x298, 0x3, 0x2, 0x2, 0x2, 0x2a1, 0x299, 
       0x3, 0x2, 0x2, 0x2, 0x2a1, 0x29a, 0x3, 0x2, 0x2, 0x2, 0x2a1, 0x29b, 
       0x3, 0x2, 0x2, 0x2, 0x2a1, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x2a2, 0x2ac, 
       0x3, 0x2, 0x2, 0x2, 0x2a3, 0x2a4, 0xc, 0x4, 0x2, 0x2, 0x2a4, 0x2a5, 
       0x7, 0x62, 0x2, 0x2, 0x2a5, 0x2a6, 0x5, 0x74, 0x3b, 0x2, 0x2a6, 0x2a7, 
       0x7, 0x63, 0x2, 0x2, 0x2a7, 0x2ab, 0x3, 0x2, 0x2, 0x2, 0x2a8, 0x2a9, 
       0xc, 0x3, 0x2, 0x2, 0x2a9, 0x2ab, 0x5, 0x84, 0x43, 0x2, 0x2aa, 0x2a3, 
       0x3, 0x2, 0x2, 0x2, 0x2aa, 0x2a8, 0x3, 0x2, 0x2, 0x2, 0x2ab, 0x2ae, 
       0x3, 0x2, 0x2, 0x2, 0x2ac, 0x2aa, 0x3, 0x2, 0x2, 0x2, 0x2ac, 0x2ad, 
       0x3, 0x2, 0x2, 0x2, 0x2ad, 0x83, 0x3, 0x2, 0x2, 0x2, 0x2ae, 0x2ac, 
       0x3, 0x2, 0x2, 0x2, 0x2af, 0x2b0, 0x9, 0xd, 0x2, 0x2, 0x2b0, 0x85, 
       0x3, 0x2, 0x2, 0x2, 0x2b1, 0x2b4, 0x5, 0x88, 0x45, 0x2, 0x2b2, 0x2b4, 
       0x5, 0x8a, 0x46, 0x2, 0x2b3, 0x2b1, 0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2b2, 
       0x3, 0x2, 0x2, 0x2, 0x2b4, 0x2b5, 0x3, 0x2, 0x2, 0x2, 0x2b5, 0x2b6, 
       0x7, 0x66, 0x2, 0x2, 0x2b6, 0x2b7, 0x5, 0x8c, 0x47, 0x2, 0x2b7, 0x2b8, 
       0x7, 0x67, 0x2, 0x2, 0x2b8, 0x87, 0x3, 0x2, 0x2, 0x2, 0x2b9, 0x2ba, 
       0x9, 0xe, 0x2, 0x2, 0x2ba, 0x89, 0x3, 0x2, 0x2, 0x2, 0x2bb, 0x2bc, 
       0x5, 0x32, 0x1a, 0x2, 0x2bc, 0x8b, 0x3, 0x2, 0x2, 0x2, 0x2bd, 0x2be, 
       0x5, 0x74, 0x3b, 0x2, 0x2be, 0x2bf, 0x7, 0x6b, 0x2, 0x2, 0x2bf, 0x2c1, 
       0x3, 0x2, 0x2, 0x2, 0x2c0, 0x2bd, 0x3, 0x2, 0x2, 0x2, 0x2c1, 0x2c4, 
       0x3, 0x2, 0x2, 0x2, 0x2c2, 0x2c0, 0x3, 0x2, 0x2, 0x2, 0x2c2, 0x2c3, 
       0x3, 0x2, 0x2, 0x2, 0x2c3, 0x2c5, 0x3, 0x2, 0x2, 0x2, 0x2c4, 0x2c2, 
       0x3, 0x2, 0x2, 0x2, 0x2c5, 0x2c6, 0x5, 0x74, 0x3b, 0x2, 0x2c6, 0x8d, 
       0x3, 0x2, 0x2, 0x2, 0x2c7, 0x2c8, 0x8, 0x48, 0x1, 0x2, 0x2c8, 0x2cb, 
       0x5, 0x98, 0x4d, 0x2, 0x2c9, 0x2cb, 0x5, 0x90, 0x49, 0x2, 0x2ca, 
       0x2c7, 0x3, 0x2, 0x2, 0x2, 0x2ca, 0x2c9, 0x3, 0x2, 0x2, 0x2, 0x2cb, 
       0x2d2, 0x3, 0x2, 0x2, 0x2, 0x2cc, 0x2cd, 0xc, 0x3, 0x2, 0x2, 0x2cd, 
       0x2ce, 0x5, 0x70, 0x39, 0x2, 0x2ce, 0x2cf, 0x5, 0x90, 0x49, 0x2, 
       0x2cf, 0x2d1, 0x3, 0x2, 0x2, 0x2, 0x2d0, 0x2cc, 0x3, 0x2, 0x2, 0x2, 
       0x2d1, 0x2d4, 0x3, 0x2, 0x2, 0x2, 0x2d2, 0x2d0, 0x3, 0x2, 0x2, 0x2, 
       0x2d2, 0x2d3, 0x3, 0x2, 0x2, 0x2, 0x2d3, 0x8f, 0x3, 0x2, 0x2, 0x2, 
       0x2d4, 0x2d2, 0x3, 0x2, 0x2, 0x2, 0x2d5, 0x2db, 0x5, 0x74, 0x3b, 
       0x2, 0x2d6, 0x2d7, 0x5, 0x74, 0x3b, 0x2, 0x2d7, 0x2d8, 0x5, 0x6e, 
       0x38, 0x2, 0x2d8, 0x2d9, 0x5, 0x74, 0x3b, 0x2, 0x2d9, 0x2db, 0x3, 
       0x2, 0x2, 0x2, 0x2da, 0x2d5, 0x3, 0x2, 0x2, 0x2, 0x2da, 0x2d6, 0x3, 
       0x2, 0x2, 0x2, 0x2db, 0x91, 0x3, 0x2, 0x2, 0x2, 0x2dc, 0x2dd, 0x7, 
       0x6c, 0x2, 0x2, 0x2dd, 0x2de, 0x5, 0x74, 0x3b, 0x2, 0x2de, 0x93, 
       0x3, 0x2, 0x2, 0x2, 0x2df, 0x2e0, 0x9, 0xf, 0x2, 0x2, 0x2e0, 0x95, 
       0x3, 0x2, 0x2, 0x2, 0x2e1, 0x2e2, 0x7, 0x73, 0x2, 0x2, 0x2e2, 0x2e3, 
       0x5, 0x92, 0x4a, 0x2, 0x2e3, 0x2e4, 0x7, 0x6b, 0x2, 0x2, 0x2e4, 0x2e6, 
       0x3, 0x2, 0x2, 0x2, 0x2e5, 0x2e1, 0x3, 0x2, 0x2, 0x2, 0x2e6, 0x2e9, 
       0x3, 0x2, 0x2, 0x2, 0x2e7, 0x2e5, 0x3, 0x2, 0x2, 0x2, 0x2e7, 0x2e8, 
       0x3, 0x2, 0x2, 0x2, 0x2e8, 0x2ea, 0x3, 0x2, 0x2, 0x2, 0x2e9, 0x2e7, 
       0x3, 0x2, 0x2, 0x2, 0x2ea, 0x2eb, 0x7, 0x73, 0x2, 0x2, 0x2eb, 0x2ec, 
       0x5, 0x92, 0x4a, 0x2, 0x2ec, 0x97, 0x3, 0x2, 0x2, 0x2, 0x2ed, 0x2ee, 
       0x7, 0x73, 0x2, 0x2, 0x2ee, 0x2ef, 0x7, 0x4b, 0x2, 0x2, 0x2ef, 0x2f0, 
       0x5, 0x9a, 0x4e, 0x2, 0x2f0, 0x99, 0x3, 0x2, 0x2, 0x2, 0x2f1, 0x2f2, 
       0x7, 0x64, 0x2, 0x2, 0x2f2, 0x2f3, 0x5, 0x8c, 0x47, 0x2, 0x2f3, 0x2f4, 
       0x7, 0x65, 0x2, 0x2, 0x2f4, 0x2f8, 0x3, 0x2, 0x2, 0x2, 0x2f5, 0x2f8, 
       0x5, 0x4e, 0x28, 0x2, 0x2f6, 0x2f8, 0x7, 0x73, 0x2, 0x2, 0x2f7, 0x2f1, 
       0x3, 0x2, 0x2, 0x2, 0x2f7, 0x2f5, 0x3, 0x2, 0x2, 0x2, 0x2f7, 0x2f6, 
       0x3, 0x2, 0x2, 0x2, 0x2f8, 0x9b, 0x3, 0x2, 0x2, 0x2, 0x2f9, 0x303, 
       0x5, 0xc, 0x7, 0x2, 0x2fa, 0x2fe, 0x7, 0x64, 0x2, 0x2, 0x2fb, 0x2fd, 
       0x5, 0xc, 0x7, 0x2, 0x2fc, 0x2fb, 0x3, 0x2, 0x2, 0x2, 0x2fd, 0x300, 
       0x3, 0x2, 0x2, 0x2, 0x2fe, 0x2fc, 0x3, 0x2, 0x2, 0x2, 0x2fe, 0x2ff, 
       0x3, 0x2, 0x2, 0x2, 0x2ff, 0x301, 0x3, 0x2, 0x2, 0x2, 0x300, 0x2fe, 
       0x3, 0x2, 0x2, 0x2, 0x301, 0x303, 0x7, 0x65, 0x2, 0x2, 0x302, 0x2f9, 
       0x3, 0x2, 0x2, 0x2, 0x302, 0x2fa, 0x3, 0x2, 0x2, 0x2, 0x303, 0x9d, 
       0x3, 0x2, 0x2, 0x2, 0x304, 0x305, 0x7, 0x4c, 0x2, 0x2, 0x305, 0x306, 
       0x7, 0x66, 0x2, 0x2, 0x306, 0x307, 0x5, 0x8e, 0x48, 0x2, 0x307, 0x308, 
       0x7, 0x67, 0x2, 0x2, 0x308, 0x30b, 0x5, 0x9c, 0x4f, 0x2, 0x309, 0x30a, 
       0x7, 0x4d, 0x2, 0x2, 0x30a, 0x30c, 0x5, 0x9c, 0x4f, 0x2, 0x30b, 0x309, 
       0x3, 0x2, 0x2, 0x2, 0x30b, 0x30c, 0x3, 0x2, 0x2, 0x2, 0x30c, 0x9f, 
       0x3, 0x2, 0x2, 0x2, 0x30d, 0x30e, 0x7, 0x4e, 0x2, 0x2, 0x30e, 0x315, 
       0x5, 0x98, 0x4d, 0x2, 0x30f, 0x310, 0x7, 0x4f, 0x2, 0x2, 0x310, 0x311, 
       0x7, 0x66, 0x2, 0x2, 0x311, 0x312, 0x5, 0x8e, 0x48, 0x2, 0x312, 0x313, 
       0x7, 0x67, 0x2, 0x2, 0x313, 0x315, 0x3, 0x2, 0x2, 0x2, 0x314, 0x30d, 
       0x3, 0x2, 0x2, 0x2, 0x314, 0x30f, 0x3, 0x2, 0x2, 0x2, 0x315, 0xa1, 
       0x3, 0x2, 0x2, 0x2, 0x316, 0x317, 0x5, 0xa0, 0x51, 0x2, 0x317, 0x318, 
       0x5, 0x9c, 0x4f, 0x2, 0x318, 0xa3, 0x3, 0x2, 0x2, 0x2, 0x319, 0x31a, 
       0x7, 0x4e, 0x2, 0x2, 0x31a, 0x31b, 0x7, 0x66, 0x2, 0x2, 0x31b, 0x31c, 
       0x5, 0x32, 0x1a, 0x2, 0x31c, 0x31d, 0x7, 0x73, 0x2, 0x2, 0x31d, 0x31e, 
       0x5, 0x92, 0x4a, 0x2, 0x31e, 0x31f, 0x7, 0x69, 0x2, 0x2, 0x31f, 0x320, 
       0x5, 0x8e, 0x48, 0x2, 0x320, 0x321, 0x7, 0x69, 0x2, 0x2, 0x321, 0x322, 
       0x5, 0x74, 0x3b, 0x2, 0x322, 0x323, 0x7, 0x67, 0x2, 0x2, 0x323, 0x324, 
       0x7, 0x64, 0x2, 0x2, 0x324, 0x325, 0x5, 0x9c, 0x4f, 0x2, 0x325, 0x326, 
       0x7, 0x65, 0x2, 0x2, 0x326, 0x336, 0x3, 0x2, 0x2, 0x2, 0x327, 0x328, 
       0x7, 0x4e, 0x2, 0x2, 0x328, 0x329, 0x7, 0x66, 0x2, 0x2, 0x329, 0x32a, 
       0x5, 0x32, 0x1a, 0x2, 0x32a, 0x32b, 0x7, 0x73, 0x2, 0x2, 0x32b, 0x32c, 
       0x7, 0x68, 0x2, 0x2, 0x32c, 0x32d, 0x7, 0x50, 0x2, 0x2, 0x32d, 0x32e, 
       0x7, 0x66, 0x2, 0x2, 0x32e, 0x32f, 0x5, 0x74, 0x3b, 0x2, 0x32f, 0x330, 
       0x7, 0x67, 0x2, 0x2, 0x330, 0x331, 0x7, 0x67, 0x2, 0x2, 0x331, 0x332, 
       0x7, 0x64, 0x2, 0x2, 0x332, 0x333, 0x5, 0x9c, 0x4f, 0x2, 0x333, 0x334, 
       0x7, 0x65, 0x2, 0x2, 0x334, 0x336, 0x3, 0x2, 0x2, 0x2, 0x335, 0x319, 
       0x3, 0x2, 0x2, 0x2, 0x335, 0x327, 0x3, 0x2, 0x2, 0x2, 0x336, 0xa5, 
       0x3, 0x2, 0x2, 0x2, 0x337, 0x338, 0x5, 0xa8, 0x55, 0x2, 0x338, 0x339, 
       0x7, 0x69, 0x2, 0x2, 0x339, 0xa7, 0x3, 0x2, 0x2, 0x2, 0x33a, 0x33b, 
       0x9, 0x10, 0x2, 0x2, 0x33b, 0xa9, 0x3, 0x2, 0x2, 0x2, 0x33c, 0x33d, 
       0x7, 0x54, 0x2, 0x2, 0x33d, 0x343, 0x7, 0x73, 0x2, 0x2, 0x33e, 0x340, 
       0x7, 0x66, 0x2, 0x2, 0x33f, 0x341, 0x5, 0x40, 0x21, 0x2, 0x340, 0x33f, 
       0x3, 0x2, 0x2, 0x2, 0x340, 0x341, 0x3, 0x2, 0x2, 0x2, 0x341, 0x342, 
       0x3, 0x2, 0x2, 0x2, 0x342, 0x344, 0x7, 0x67, 0x2, 0x2, 0x343, 0x33e, 
       0x3, 0x2, 0x2, 0x2, 0x343, 0x344, 0x3, 0x2, 0x2, 0x2, 0x344, 0x346, 
       0x3, 0x2, 0x2, 0x2, 0x345, 0x347, 0x5, 0x18, 0xd, 0x2, 0x346, 0x345, 
       0x3, 0x2, 0x2, 0x2, 0x346, 0x347, 0x3, 0x2, 0x2, 0x2, 0x347, 0x349, 
       0x3, 0x2, 0x2, 0x2, 0x348, 0x34a, 0x5, 0x32, 0x1a, 0x2, 0x349, 0x348, 
       0x3, 0x2, 0x2, 0x2, 0x349, 0x34a, 0x3, 0x2, 0x2, 0x2, 0x34a, 0x34b, 
       0x3, 0x2, 0x2, 0x2, 0x34b, 0x34c, 0x7, 0x69, 0x2, 0x2, 0x34c, 0xab, 
       0x3, 0x2, 0x2, 0x2, 0x34d, 0x34e, 0x7, 0x73, 0x2, 0x2, 0x34e, 0x350, 
       0x7, 0x66, 0x2, 0x2, 0x34f, 0x351, 0x5, 0x8c, 0x47, 0x2, 0x350, 0x34f, 
       0x3, 0x2, 0x2, 0x2, 0x350, 0x351, 0x3, 0x2, 0x2, 0x2, 0x351, 0x352, 
       0x3, 0x2, 0x2, 0x2, 0x352, 0x353, 0x7, 0x67, 0x2, 0x2, 0x353, 0xad, 
       0x3, 0x2, 0x2, 0x2, 0x354, 0x355, 0x7, 0x55, 0x2, 0x2, 0x355, 0x35b, 
       0x7, 0x73, 0x2, 0x2, 0x356, 0x358, 0x7, 0x66, 0x2, 0x2, 0x357, 0x359, 
       0x5, 0x44, 0x23, 0x2, 0x358, 0x357, 0x3, 0x2, 0x2, 0x2, 0x358, 0x359, 
       0x3, 0x2, 0x2, 0x2, 0x359, 0x35a, 0x3, 0x2, 0x2, 0x2, 0x35a, 0x35c, 
       0x7, 0x67, 0x2, 0x2, 0x35b, 0x356, 0x3, 0x2, 0x2, 0x2, 0x35b, 0x35c, 
       0x3, 0x2, 0x2, 0x2, 0x35c, 0x35e, 0x3, 0x2, 0x2, 0x2, 0x35d, 0x35f, 
       0x5, 0x28, 0x15, 0x2, 0x35e, 0x35d, 0x3, 0x2, 0x2, 0x2, 0x35e, 0x35f, 
       0x3, 0x2, 0x2, 0x2, 0x35f, 0x361, 0x3, 0x2, 0x2, 0x2, 0x360, 0x362, 
       0x5, 0x18, 0xd, 0x2, 0x361, 0x360, 0x3, 0x2, 0x2, 0x2, 0x361, 0x362, 
       0x3, 0x2, 0x2, 0x2, 0x362, 0x363, 0x3, 0x2, 0x2, 0x2, 0x363, 0x364, 
       0x5, 0xb2, 0x5a, 0x2, 0x364, 0xaf, 0x3, 0x2, 0x2, 0x2, 0x365, 0x366, 
       0x7, 0x56, 0x2, 0x2, 0x366, 0x367, 0x5, 0xc, 0x7, 0x2, 0x367, 0xb1, 
       0x3, 0x2, 0x2, 0x2, 0x368, 0x36c, 0x7, 0x64, 0x2, 0x2, 0x369, 0x36b, 
       0x5, 0xc, 0x7, 0x2, 0x36a, 0x369, 0x3, 0x2, 0x2, 0x2, 0x36b, 0x36e, 
       0x3, 0x2, 0x2, 0x2, 0x36c, 0x36a, 0x3, 0x2, 0x2, 0x2, 0x36c, 0x36d, 
       0x3, 0x2, 0x2, 0x2, 0x36d, 0x370, 0x3, 0x2, 0x2, 0x2, 0x36e, 0x36c, 
       0x3, 0x2, 0x2, 0x2, 0x36f, 0x371, 0x5, 0xb0, 0x59, 0x2, 0x370, 0x36f, 
       0x3, 0x2, 0x2, 0x2, 0x370, 0x371, 0x3, 0x2, 0x2, 0x2, 0x371, 0x372, 
       0x3, 0x2, 0x2, 0x2, 0x372, 0x373, 0x7, 0x65, 0x2, 0x2, 0x373, 0xb3, 
       0x3, 0x2, 0x2, 0x2, 0x374, 0x375, 0x7, 0x73, 0x2, 0x2, 0x375, 0x377, 
       0x7, 0x66, 0x2, 0x2, 0x376, 0x378, 0x5, 0x8c, 0x47, 0x2, 0x377, 0x376, 
       0x3, 0x2, 0x2, 0x2, 0x377, 0x378, 0x3, 0x2, 0x2, 0x2, 0x378, 0x379, 
       0x3, 0x2, 0x2, 0x2, 0x379, 0x37a, 0x7, 0x67, 0x2, 0x2, 0x37a, 0x37b, 
       0x3, 0x2, 0x2, 0x2, 0x37b, 0x37c, 0x5, 0x8c, 0x47, 0x2, 0x37c, 0xb5, 
       0x3, 0x2, 0x2, 0x2, 0x37d, 0x37e, 0x7, 0x57, 0x2, 0x2, 0x37e, 0x382, 
       0x7, 0x64, 0x2, 0x2, 0x37f, 0x381, 0x5, 0xc, 0x7, 0x2, 0x380, 0x37f, 
       0x3, 0x2, 0x2, 0x2, 0x381, 0x384, 0x3, 0x2, 0x2, 0x2, 0x382, 0x380, 
       0x3, 0x2, 0x2, 0x2, 0x382, 0x383, 0x3, 0x2, 0x2, 0x2, 0x383, 0x385, 
       0x3, 0x2, 0x2, 0x2, 0x384, 0x382, 0x3, 0x2, 0x2, 0x2, 0x385, 0x386, 
       0x7, 0x65, 0x2, 0x2, 0x386, 0xb7, 0x3, 0x2, 0x2, 0x2, 0x387, 0x38d, 
       0x7, 0x58, 0x2, 0x2, 0x388, 0x38a, 0x7, 0x59, 0x2, 0x2, 0x389, 0x38b, 
       0x7, 0x72, 0x2, 0x2, 0x38a, 0x389, 0x3, 0x2, 0x2, 0x2, 0x38a, 0x38b, 
       0x3, 0x2, 0x2, 0x2, 0x38b, 0x38d, 0x3, 0x2, 0x2, 0x2, 0x38c, 0x387, 
       0x3, 0x2, 0x2, 0x2, 0x38c, 0x388, 0x3, 0x2, 0x2, 0x2, 0x38d, 0xb9, 
       0x3, 0x2, 0x2, 0x2, 0x38e, 0x38f, 0x7, 0x5a, 0x2, 0x2, 0x38f, 0x390, 
       0x7, 0x73, 0x2, 0x2, 0x390, 0x395, 0x5, 0x54, 0x2b, 0x2, 0x391, 0x392, 
       0x7, 0x5b, 0x2, 0x2, 0x392, 0x393, 0x7, 0x75, 0x2, 0x2, 0x393, 0x395, 
       0x5, 0x54, 0x2b, 0x2, 0x394, 0x38e, 0x3, 0x2, 0x2, 0x2, 0x394, 0x391, 
       0x3, 0x2, 0x2, 0x2, 0x395, 0xbb, 0x3, 0x2, 0x2, 0x2, 0x396, 0x399, 
       0x5, 0xbe, 0x60, 0x2, 0x397, 0x399, 0x7, 0x5c, 0x2, 0x2, 0x398, 0x396, 
       0x3, 0x2, 0x2, 0x2, 0x398, 0x397, 0x3, 0x2, 0x2, 0x2, 0x399, 0xbd, 
       0x3, 0x2, 0x2, 0x2, 0x39a, 0x3a3, 0x7, 0x75, 0x2, 0x2, 0x39b, 0x39c, 
       0x7, 0x40, 0x2, 0x2, 0x39c, 0x39f, 0x7, 0x66, 0x2, 0x2, 0x39d, 0x3a0, 
       0x7, 0x73, 0x2, 0x2, 0x39e, 0x3a0, 0x5, 0x54, 0x2b, 0x2, 0x39f, 0x39d, 
       0x3, 0x2, 0x2, 0x2, 0x39f, 0x39e, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x3a1, 
       0x3, 0x2, 0x2, 0x2, 0x3a1, 0x3a3, 0x7, 0x67, 0x2, 0x2, 0x3a2, 0x39a, 
       0x3, 0x2, 0x2, 0x2, 0x3a2, 0x39b, 0x3, 0x2, 0x2, 0x2, 0x3a3, 0xbf, 
       0x3, 0x2, 0x2, 0x2, 0x3a4, 0x3a5, 0x9, 0x11, 0x2, 0x2, 0x3a5, 0xc1, 
       0x3, 0x2, 0x2, 0x2, 0x3a6, 0x3ac, 0x5, 0xc0, 0x61, 0x2, 0x3a7, 0x3a9, 
       0x7, 0x66, 0x2, 0x2, 0x3a8, 0x3aa, 0x5, 0x8c, 0x47, 0x2, 0x3a9, 0x3a8, 
       0x3, 0x2, 0x2, 0x2, 0x3a9, 0x3aa, 0x3, 0x2, 0x2, 0x2, 0x3aa, 0x3ab, 
       0x3, 0x2, 0x2, 0x2, 0x3ab, 0x3ad, 0x7, 0x67, 0x2, 0x2, 0x3ac, 0x3a7, 
       0x3, 0x2, 0x2, 0x2, 0x3ac, 0x3ad, 0x3, 0x2, 0x2, 0x2, 0x3ad, 0x3ae, 
       0x3, 0x2, 0x2, 0x2, 0x3ae, 0x3af, 0x5, 0x1a, 0xe, 0x2, 0x3af, 0x3b0, 
       0x5, 0x4a, 0x26, 0x2, 0x3b0, 0xc3, 0x3, 0x2, 0x2, 0x2, 0x3b1, 0x3b2, 
       0x5, 0xc2, 0x62, 0x2, 0x3b2, 0x3b3, 0x7, 0x69, 0x2, 0x2, 0x3b3, 0x3b6, 
       0x3, 0x2, 0x2, 0x2, 0x3b4, 0x3b6, 0x5, 0xba, 0x5e, 0x2, 0x3b5, 0x3b1, 
       0x3, 0x2, 0x2, 0x2, 0x3b5, 0x3b4, 0x3, 0x2, 0x2, 0x2, 0x3b6, 0xc5, 
       0x3, 0x2, 0x2, 0x2, 0x3b7, 0x3ba, 0x5, 0xc8, 0x65, 0x2, 0x3b8, 0x3ba, 
       0x5, 0xca, 0x66, 0x2, 0x3b9, 0x3b7, 0x3, 0x2, 0x2, 0x2, 0x3b9, 0x3b8, 
       0x3, 0x2, 0x2, 0x2, 0x3ba, 0xc7, 0x3, 0x2, 0x2, 0x2, 0x3bb, 0x3bc, 
       0x7, 0x5f, 0x2, 0x2, 0x3bc, 0x3bd, 0x5, 0xcc, 0x67, 0x2, 0x3bd, 0x3be, 
       0x7, 0x69, 0x2, 0x2, 0x3be, 0xc9, 0x3, 0x2, 0x2, 0x2, 0x3bf, 0x3c0, 
       0x7, 0x60, 0x2, 0x2, 0x3c0, 0x3c6, 0x7, 0x73, 0x2, 0x2, 0x3c1, 0x3c3, 
       0x7, 0x66, 0x2, 0x2, 0x3c2, 0x3c4, 0x5, 0xce, 0x68, 0x2, 0x3c3, 0x3c2, 
       0x3, 0x2, 0x2, 0x2, 0x3c3, 0x3c4, 0x3, 0x2, 0x2, 0x2, 0x3c4, 0x3c5, 
       0x3, 0x2, 0x2, 0x2, 0x3c5, 0x3c7, 0x7, 0x67, 0x2, 0x2, 0x3c6, 0x3c1, 
       0x3, 0x2, 0x2, 0x2, 0x3c6, 0x3c7, 0x3, 0x2, 0x2, 0x2, 0x3c7, 0x3c8, 
       0x3, 0x2, 0x2, 0x2, 0x3c8, 0x3ca, 0x5, 0x1e, 0x10, 0x2, 0x3c9, 0x3cb, 
       0x5, 0x18, 0xd, 0x2, 0x3ca, 0x3c9, 0x3, 0x2, 0x2, 0x2, 0x3ca, 0x3cb, 
       0x3, 0x2, 0x2, 0x2, 0x3cb, 0x3cc, 0x3, 0x2, 0x2, 0x2, 0x3cc, 0x3d0, 
       0x7, 0x64, 0x2, 0x2, 0x3cd, 0x3cf, 0xb, 0x2, 0x2, 0x2, 0x3ce, 0x3cd, 
       0x3, 0x2, 0x2, 0x2, 0x3cf, 0x3d2, 0x3, 0x2, 0x2, 0x2, 0x3d0, 0x3d1, 
       0x3, 0x2, 0x2, 0x2, 0x3d0, 0x3ce, 0x3, 0x2, 0x2, 0x2, 0x3d1, 0x3d3, 
       0x3, 0x2, 0x2, 0x2, 0x3d2, 0x3d0, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3d4, 
       0x7, 0x65, 0x2, 0x2, 0x3d4, 0xcb, 0x3, 0x2, 0x2, 0x2, 0x3d5, 0x3d6, 
       0x9, 0x12, 0x2, 0x2, 0x3d6, 0xcd, 0x3, 0x2, 0x2, 0x2, 0x3d7, 0x3da, 
       0x5, 0x44, 0x23, 0x2, 0x3d8, 0x3da, 0x5, 0x8c, 0x47, 0x2, 0x3d9, 
       0x3d7, 0x3, 0x2, 0x2, 0x2, 0x3d9, 0x3d8, 0x3, 0x2, 0x2, 0x2, 0x3da, 
       0xcf, 0x3, 0x2, 0x2, 0x2, 0x62, 0xd3, 0xd5, 0xd9, 0xde, 0xef, 0xfb, 
       0x108, 0x110, 0x114, 0x129, 0x138, 0x141, 0x152, 0x15d, 0x15f, 0x168, 
       0x16e, 0x173, 0x178, 0x17e, 0x185, 0x192, 0x1a5, 0x1a7, 0x1ae, 0x1b6, 
       0x1c1, 0x1c9, 0x1cd, 0x1d1, 0x1dc, 0x1df, 0x1e6, 0x1e8, 0x1f5, 0x1f9, 
       0x1ff, 0x205, 0x212, 0x218, 0x224, 0x22b, 0x22e, 0x239, 0x248, 0x24f, 
       0x25a, 0x265, 0x270, 0x27b, 0x281, 0x287, 0x28b, 0x2a1, 0x2aa, 0x2ac, 
       0x2b3, 0x2c2, 0x2ca, 0x2d2, 0x2da, 0x2e7, 0x2f7, 0x2fe, 0x302, 0x30b, 
       0x314, 0x335, 0x340, 0x343, 0x346, 0x349, 0x350, 0x358, 0x35b, 0x35e, 
       0x361, 0x36c, 0x370, 0x377, 0x382, 0x38a, 0x38c, 0x394, 0x398, 0x39f, 
       0x3a2, 0x3a9, 0x3ac, 0x3b5, 0x3b9, 0x3c3, 0x3c6, 0x3ca, 0x3d0, 0x3d9, 
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

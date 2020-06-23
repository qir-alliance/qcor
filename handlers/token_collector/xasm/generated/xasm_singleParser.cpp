
// Generated from xasm_single.g4 by ANTLR 4.7.2


#include "xasm_singleListener.h"
#include "xasm_singleVisitor.h"

#include "xasm_singleParser.h"


using namespace antlrcpp;
using namespace xasm;
using namespace antlr4;

xasm_singleParser::xasm_singleParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

xasm_singleParser::~xasm_singleParser() {
  delete _interpreter;
}

std::string xasm_singleParser::getGrammarFileName() const {
  return "xasm_single.g4";
}

const std::vector<std::string>& xasm_singleParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& xasm_singleParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- LineContext ------------------------------------------------------------------

xasm_singleParser::LineContext::LineContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

xasm_singleParser::StatementContext* xasm_singleParser::LineContext::statement() {
  return getRuleContext<xasm_singleParser::StatementContext>(0);
}

xasm_singleParser::CommentContext* xasm_singleParser::LineContext::comment() {
  return getRuleContext<xasm_singleParser::CommentContext>(0);
}


size_t xasm_singleParser::LineContext::getRuleIndex() const {
  return xasm_singleParser::RuleLine;
}

void xasm_singleParser::LineContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLine(this);
}

void xasm_singleParser::LineContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLine(this);
}


antlrcpp::Any xasm_singleParser::LineContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitLine(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::LineContext* xasm_singleParser::line() {
  LineContext *_localctx = _tracker.createInstance<LineContext>(_ctx, getState());
  enterRule(_localctx, 0, xasm_singleParser::RuleLine);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(28);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case xasm_singleParser::T__0:
      case xasm_singleParser::T__3:
      case xasm_singleParser::T__7:
      case xasm_singleParser::T__9:
      case xasm_singleParser::T__13:
      case xasm_singleParser::T__22:
      case xasm_singleParser::T__30:
      case xasm_singleParser::T__31:
      case xasm_singleParser::T__32:
      case xasm_singleParser::T__33:
      case xasm_singleParser::T__34:
      case xasm_singleParser::T__35:
      case xasm_singleParser::T__36:
      case xasm_singleParser::ID:
      case xasm_singleParser::REAL:
      case xasm_singleParser::INT:
      case xasm_singleParser::STRING: {
        enterOuterAlt(_localctx, 1);
        setState(26);
        statement();
        break;
      }

      case xasm_singleParser::COMMENT: {
        enterOuterAlt(_localctx, 2);
        setState(27);
        comment();
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

xasm_singleParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

xasm_singleParser::QinstContext* xasm_singleParser::StatementContext::qinst() {
  return getRuleContext<xasm_singleParser::QinstContext>(0);
}

xasm_singleParser::CinstContext* xasm_singleParser::StatementContext::cinst() {
  return getRuleContext<xasm_singleParser::CinstContext>(0);
}


size_t xasm_singleParser::StatementContext::getRuleIndex() const {
  return xasm_singleParser::RuleStatement;
}

void xasm_singleParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void xasm_singleParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}


antlrcpp::Any xasm_singleParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::StatementContext* xasm_singleParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 2, xasm_singleParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(32);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(30);
      qinst();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(31);
      cinst();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CommentContext ------------------------------------------------------------------

xasm_singleParser::CommentContext::CommentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* xasm_singleParser::CommentContext::COMMENT() {
  return getToken(xasm_singleParser::COMMENT, 0);
}


size_t xasm_singleParser::CommentContext::getRuleIndex() const {
  return xasm_singleParser::RuleComment;
}

void xasm_singleParser::CommentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComment(this);
}

void xasm_singleParser::CommentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComment(this);
}


antlrcpp::Any xasm_singleParser::CommentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitComment(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::CommentContext* xasm_singleParser::comment() {
  CommentContext *_localctx = _tracker.createInstance<CommentContext>(_ctx, getState());
  enterRule(_localctx, 4, xasm_singleParser::RuleComment);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(34);
    match(xasm_singleParser::COMMENT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QinstContext ------------------------------------------------------------------

xasm_singleParser::QinstContext::QinstContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

xasm_singleParser::ExplistContext* xasm_singleParser::QinstContext::explist() {
  return getRuleContext<xasm_singleParser::ExplistContext>(0);
}

xasm_singleParser::IdContext* xasm_singleParser::QinstContext::id() {
  return getRuleContext<xasm_singleParser::IdContext>(0);
}


size_t xasm_singleParser::QinstContext::getRuleIndex() const {
  return xasm_singleParser::RuleQinst;
}

void xasm_singleParser::QinstContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQinst(this);
}

void xasm_singleParser::QinstContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQinst(this);
}


antlrcpp::Any xasm_singleParser::QinstContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitQinst(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::QinstContext* xasm_singleParser::qinst() {
  QinstContext *_localctx = _tracker.createInstance<QinstContext>(_ctx, getState());
  enterRule(_localctx, 6, xasm_singleParser::RuleQinst);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(36);
    dynamic_cast<QinstContext *>(_localctx)->inst_name = id();
    setState(37);
    match(xasm_singleParser::T__0);
    setState(38);
    explist();
    setState(39);
    match(xasm_singleParser::T__1);
    setState(40);
    match(xasm_singleParser::T__2);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CinstContext ------------------------------------------------------------------

xasm_singleParser::CinstContext::CinstContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

xasm_singleParser::Cpp_typeContext* xasm_singleParser::CinstContext::cpp_type() {
  return getRuleContext<xasm_singleParser::Cpp_typeContext>(0);
}

std::vector<xasm_singleParser::ExpContext *> xasm_singleParser::CinstContext::exp() {
  return getRuleContexts<xasm_singleParser::ExpContext>();
}

xasm_singleParser::ExpContext* xasm_singleParser::CinstContext::exp(size_t i) {
  return getRuleContext<xasm_singleParser::ExpContext>(i);
}

xasm_singleParser::CompareContext* xasm_singleParser::CinstContext::compare() {
  return getRuleContext<xasm_singleParser::CompareContext>(0);
}

xasm_singleParser::ExplistContext* xasm_singleParser::CinstContext::explist() {
  return getRuleContext<xasm_singleParser::ExplistContext>(0);
}


size_t xasm_singleParser::CinstContext::getRuleIndex() const {
  return xasm_singleParser::RuleCinst;
}

void xasm_singleParser::CinstContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCinst(this);
}

void xasm_singleParser::CinstContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCinst(this);
}


antlrcpp::Any xasm_singleParser::CinstContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitCinst(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::CinstContext* xasm_singleParser::cinst() {
  CinstContext *_localctx = _tracker.createInstance<CinstContext>(_ctx, getState());
  enterRule(_localctx, 8, xasm_singleParser::RuleCinst);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(110);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(43);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == xasm_singleParser::T__3) {
        setState(42);
        match(xasm_singleParser::T__3);
      }
      setState(45);
      dynamic_cast<CinstContext *>(_localctx)->type_name = cpp_type();
      setState(46);
      dynamic_cast<CinstContext *>(_localctx)->var_name = exp(0);
      setState(49);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == xasm_singleParser::T__4) {
        setState(47);
        match(xasm_singleParser::T__4);
        setState(48);
        dynamic_cast<CinstContext *>(_localctx)->var_value = exp(0);
      }
      setState(51);
      match(xasm_singleParser::T__2);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(53);
      exp(0);
      setState(54);
      match(xasm_singleParser::T__5);
      setState(55);
      match(xasm_singleParser::T__2);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(57);
      exp(0);
      setState(58);
      match(xasm_singleParser::T__6);
      setState(59);
      match(xasm_singleParser::T__2);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(61);
      match(xasm_singleParser::T__7);
      setState(62);
      match(xasm_singleParser::T__0);
      setState(63);
      cpp_type();
      setState(64);
      exp(0);
      setState(65);
      match(xasm_singleParser::T__4);
      setState(66);
      exp(0);
      setState(67);
      match(xasm_singleParser::T__2);
      setState(72);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << xasm_singleParser::T__0)
        | (1ULL << xasm_singleParser::T__22)
        | (1ULL << xasm_singleParser::T__30)
        | (1ULL << xasm_singleParser::T__31)
        | (1ULL << xasm_singleParser::T__32)
        | (1ULL << xasm_singleParser::T__33)
        | (1ULL << xasm_singleParser::T__34)
        | (1ULL << xasm_singleParser::T__35)
        | (1ULL << xasm_singleParser::T__36)
        | (1ULL << xasm_singleParser::ID)
        | (1ULL << xasm_singleParser::REAL)
        | (1ULL << xasm_singleParser::INT)
        | (1ULL << xasm_singleParser::STRING))) != 0)) {
        setState(68);
        exp(0);
        setState(69);
        compare();
        setState(70);
        exp(0);
      }
      setState(74);
      match(xasm_singleParser::T__2);
      setState(80);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case xasm_singleParser::T__0:
        case xasm_singleParser::T__22:
        case xasm_singleParser::T__30:
        case xasm_singleParser::T__31:
        case xasm_singleParser::T__32:
        case xasm_singleParser::T__33:
        case xasm_singleParser::T__34:
        case xasm_singleParser::T__35:
        case xasm_singleParser::T__36:
        case xasm_singleParser::ID:
        case xasm_singleParser::REAL:
        case xasm_singleParser::INT:
        case xasm_singleParser::STRING: {
          setState(75);
          exp(0);
          setState(76);
          _la = _input->LA(1);
          if (!(_la == xasm_singleParser::T__5

          || _la == xasm_singleParser::T__6)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          break;
        }

        case xasm_singleParser::T__5:
        case xasm_singleParser::T__6: {
          setState(78);
          _la = _input->LA(1);
          if (!(_la == xasm_singleParser::T__5

          || _la == xasm_singleParser::T__6)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(79);
          exp(0);
          break;
        }

        case xasm_singleParser::T__1: {
          break;
        }

      default:
        break;
      }
      setState(82);
      match(xasm_singleParser::T__1);
      setState(84);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == xasm_singleParser::T__8) {
        setState(83);
        match(xasm_singleParser::T__8);
      }
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(86);
      match(xasm_singleParser::T__9);
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(87);
      exp(0);
      setState(88);
      match(xasm_singleParser::T__0);
      setState(89);
      explist();
      setState(90);
      match(xasm_singleParser::T__1);
      setState(91);
      match(xasm_singleParser::T__2);
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(94);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == xasm_singleParser::T__3) {
        setState(93);
        match(xasm_singleParser::T__3);
      }
      setState(96);
      dynamic_cast<CinstContext *>(_localctx)->type_name = cpp_type();
      setState(97);
      dynamic_cast<CinstContext *>(_localctx)->var_name = exp(0);
      setState(98);
      match(xasm_singleParser::T__4);
      setState(99);
      match(xasm_singleParser::T__0);
      setState(100);
      exp(0);
      setState(101);
      match(xasm_singleParser::T__10);
      setState(102);
      exp(0);
      setState(103);
      match(xasm_singleParser::T__1);
      setState(104);
      match(xasm_singleParser::T__11);
      setState(105);
      exp(0);
      setState(106);
      match(xasm_singleParser::T__12);
      setState(107);
      exp(0);
      setState(108);
      match(xasm_singleParser::T__2);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Cpp_typeContext ------------------------------------------------------------------

xasm_singleParser::Cpp_typeContext::Cpp_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

xasm_singleParser::ExpContext* xasm_singleParser::Cpp_typeContext::exp() {
  return getRuleContext<xasm_singleParser::ExpContext>(0);
}


size_t xasm_singleParser::Cpp_typeContext::getRuleIndex() const {
  return xasm_singleParser::RuleCpp_type;
}

void xasm_singleParser::Cpp_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCpp_type(this);
}

void xasm_singleParser::Cpp_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCpp_type(this);
}


antlrcpp::Any xasm_singleParser::Cpp_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitCpp_type(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::Cpp_typeContext* xasm_singleParser::cpp_type() {
  Cpp_typeContext *_localctx = _tracker.createInstance<Cpp_typeContext>(_ctx, getState());
  enterRule(_localctx, 10, xasm_singleParser::RuleCpp_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(117);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case xasm_singleParser::T__13: {
        enterOuterAlt(_localctx, 1);
        setState(112);
        match(xasm_singleParser::T__13);
        setState(114);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == xasm_singleParser::T__14

        || _la == xasm_singleParser::T__15) {
          setState(113);
          _la = _input->LA(1);
          if (!(_la == xasm_singleParser::T__14

          || _la == xasm_singleParser::T__15)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
        }
        break;
      }

      case xasm_singleParser::T__0:
      case xasm_singleParser::T__22:
      case xasm_singleParser::T__30:
      case xasm_singleParser::T__31:
      case xasm_singleParser::T__32:
      case xasm_singleParser::T__33:
      case xasm_singleParser::T__34:
      case xasm_singleParser::T__35:
      case xasm_singleParser::T__36:
      case xasm_singleParser::ID:
      case xasm_singleParser::REAL:
      case xasm_singleParser::INT:
      case xasm_singleParser::STRING: {
        enterOuterAlt(_localctx, 2);
        setState(116);
        exp(0);
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

//----------------- CompareContext ------------------------------------------------------------------

xasm_singleParser::CompareContext::CompareContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t xasm_singleParser::CompareContext::getRuleIndex() const {
  return xasm_singleParser::RuleCompare;
}

void xasm_singleParser::CompareContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompare(this);
}

void xasm_singleParser::CompareContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompare(this);
}


antlrcpp::Any xasm_singleParser::CompareContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitCompare(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::CompareContext* xasm_singleParser::compare() {
  CompareContext *_localctx = _tracker.createInstance<CompareContext>(_ctx, getState());
  enterRule(_localctx, 12, xasm_singleParser::RuleCompare);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(119);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << xasm_singleParser::T__16)
      | (1ULL << xasm_singleParser::T__17)
      | (1ULL << xasm_singleParser::T__18)
      | (1ULL << xasm_singleParser::T__19))) != 0))) {
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

//----------------- ExplistContext ------------------------------------------------------------------

xasm_singleParser::ExplistContext::ExplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<xasm_singleParser::ExpContext *> xasm_singleParser::ExplistContext::exp() {
  return getRuleContexts<xasm_singleParser::ExpContext>();
}

xasm_singleParser::ExpContext* xasm_singleParser::ExplistContext::exp(size_t i) {
  return getRuleContext<xasm_singleParser::ExpContext>(i);
}


size_t xasm_singleParser::ExplistContext::getRuleIndex() const {
  return xasm_singleParser::RuleExplist;
}

void xasm_singleParser::ExplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExplist(this);
}

void xasm_singleParser::ExplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExplist(this);
}


antlrcpp::Any xasm_singleParser::ExplistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitExplist(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::ExplistContext* xasm_singleParser::explist() {
  ExplistContext *_localctx = _tracker.createInstance<ExplistContext>(_ctx, getState());
  enterRule(_localctx, 14, xasm_singleParser::RuleExplist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(121);
    exp(0);
    setState(126);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == xasm_singleParser::T__20) {
      setState(122);
      match(xasm_singleParser::T__20);
      setState(123);
      exp(0);
      setState(128);
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

//----------------- ExpContext ------------------------------------------------------------------

xasm_singleParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

xasm_singleParser::IdContext* xasm_singleParser::ExpContext::id() {
  return getRuleContext<xasm_singleParser::IdContext>(0);
}

std::vector<xasm_singleParser::ExpContext *> xasm_singleParser::ExpContext::exp() {
  return getRuleContexts<xasm_singleParser::ExpContext>();
}

xasm_singleParser::ExpContext* xasm_singleParser::ExpContext::exp(size_t i) {
  return getRuleContext<xasm_singleParser::ExpContext>(i);
}

xasm_singleParser::UnaryopContext* xasm_singleParser::ExpContext::unaryop() {
  return getRuleContext<xasm_singleParser::UnaryopContext>(0);
}

xasm_singleParser::StringContext* xasm_singleParser::ExpContext::string() {
  return getRuleContext<xasm_singleParser::StringContext>(0);
}

xasm_singleParser::RealContext* xasm_singleParser::ExpContext::real() {
  return getRuleContext<xasm_singleParser::RealContext>(0);
}

tree::TerminalNode* xasm_singleParser::ExpContext::INT() {
  return getToken(xasm_singleParser::INT, 0);
}

xasm_singleParser::ExplistContext* xasm_singleParser::ExpContext::explist() {
  return getRuleContext<xasm_singleParser::ExplistContext>(0);
}


size_t xasm_singleParser::ExpContext::getRuleIndex() const {
  return xasm_singleParser::RuleExp;
}

void xasm_singleParser::ExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExp(this);
}

void xasm_singleParser::ExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExp(this);
}


antlrcpp::Any xasm_singleParser::ExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitExp(this);
  else
    return visitor->visitChildren(this);
}


xasm_singleParser::ExpContext* xasm_singleParser::exp() {
   return exp(0);
}

xasm_singleParser::ExpContext* xasm_singleParser::exp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  xasm_singleParser::ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, parentState);
  xasm_singleParser::ExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 16;
  enterRecursionRule(_localctx, 16, xasm_singleParser::RuleExp, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(146);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case xasm_singleParser::ID: {
        setState(130);
        id();
        break;
      }

      case xasm_singleParser::T__22: {
        setState(131);
        match(xasm_singleParser::T__22);
        setState(132);
        exp(9);
        break;
      }

      case xasm_singleParser::T__0: {
        setState(133);
        match(xasm_singleParser::T__0);
        setState(134);
        exp(0);
        setState(135);
        match(xasm_singleParser::T__1);
        break;
      }

      case xasm_singleParser::T__31:
      case xasm_singleParser::T__32:
      case xasm_singleParser::T__33:
      case xasm_singleParser::T__34:
      case xasm_singleParser::T__35:
      case xasm_singleParser::T__36: {
        setState(137);
        unaryop();
        setState(138);
        match(xasm_singleParser::T__0);
        setState(139);
        exp(0);
        setState(140);
        match(xasm_singleParser::T__1);
        break;
      }

      case xasm_singleParser::STRING: {
        setState(142);
        string();
        break;
      }

      case xasm_singleParser::REAL: {
        setState(143);
        real();
        break;
      }

      case xasm_singleParser::INT: {
        setState(144);
        match(xasm_singleParser::INT);
        break;
      }

      case xasm_singleParser::T__30: {
        setState(145);
        match(xasm_singleParser::T__30);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(201);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(199);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(148);

          if (!(precpred(_ctx, 19))) throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(149);
          match(xasm_singleParser::T__21);
          setState(150);
          exp(20);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(151);

          if (!(precpred(_ctx, 18))) throw FailedPredicateException(this, "precpred(_ctx, 18)");
          setState(152);
          match(xasm_singleParser::T__22);
          setState(153);
          exp(19);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(154);

          if (!(precpred(_ctx, 17))) throw FailedPredicateException(this, "precpred(_ctx, 17)");
          setState(155);
          match(xasm_singleParser::T__15);
          setState(156);
          exp(18);
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(157);

          if (!(precpred(_ctx, 16))) throw FailedPredicateException(this, "precpred(_ctx, 16)");
          setState(158);
          match(xasm_singleParser::T__23);
          setState(159);
          exp(17);
          break;
        }

        case 5: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(160);

          if (!(precpred(_ctx, 15))) throw FailedPredicateException(this, "precpred(_ctx, 15)");
          setState(161);
          match(xasm_singleParser::T__24);
          setState(162);
          exp(16);
          break;
        }

        case 6: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(163);

          if (!(precpred(_ctx, 14))) throw FailedPredicateException(this, "precpred(_ctx, 14)");
          setState(164);
          match(xasm_singleParser::T__25);
          setState(165);
          exp(15);
          break;
        }

        case 7: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(166);

          if (!(precpred(_ctx, 8))) throw FailedPredicateException(this, "precpred(_ctx, 8)");
          setState(167);
          match(xasm_singleParser::T__27);
          setState(168);
          exp(9);
          break;
        }

        case 8: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(169);

          if (!(precpred(_ctx, 13))) throw FailedPredicateException(this, "precpred(_ctx, 13)");
          setState(170);
          match(xasm_singleParser::T__17);
          setState(171);
          exp(0);
          setState(172);
          match(xasm_singleParser::T__16);
          break;
        }

        case 9: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(174);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(175);
          match(xasm_singleParser::T__24);
          setState(176);
          exp(0);
          setState(177);
          match(xasm_singleParser::T__0);
          setState(178);
          explist();
          setState(179);
          match(xasm_singleParser::T__1);
          break;
        }

        case 10: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(181);

          if (!(precpred(_ctx, 11))) throw FailedPredicateException(this, "precpred(_ctx, 11)");
          setState(182);
          match(xasm_singleParser::T__26);
          setState(183);
          exp(0);
          setState(184);
          match(xasm_singleParser::T__0);
          setState(185);
          match(xasm_singleParser::T__1);
          break;
        }

        case 11: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(187);

          if (!(precpred(_ctx, 10))) throw FailedPredicateException(this, "precpred(_ctx, 10)");
          setState(188);
          match(xasm_singleParser::T__26);
          setState(189);
          exp(0);
          setState(190);
          match(xasm_singleParser::T__0);
          setState(191);
          explist();
          setState(192);
          match(xasm_singleParser::T__1);
          break;
        }

        case 12: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(194);

          if (!(precpred(_ctx, 5))) throw FailedPredicateException(this, "precpred(_ctx, 5)");
          setState(195);
          match(xasm_singleParser::T__28);
          setState(196);
          exp(0);
          setState(197);
          match(xasm_singleParser::T__29);
          break;
        }

        } 
      }
      setState(203);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- UnaryopContext ------------------------------------------------------------------

xasm_singleParser::UnaryopContext::UnaryopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t xasm_singleParser::UnaryopContext::getRuleIndex() const {
  return xasm_singleParser::RuleUnaryop;
}

void xasm_singleParser::UnaryopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryop(this);
}

void xasm_singleParser::UnaryopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryop(this);
}


antlrcpp::Any xasm_singleParser::UnaryopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitUnaryop(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::UnaryopContext* xasm_singleParser::unaryop() {
  UnaryopContext *_localctx = _tracker.createInstance<UnaryopContext>(_ctx, getState());
  enterRule(_localctx, 18, xasm_singleParser::RuleUnaryop);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(204);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << xasm_singleParser::T__31)
      | (1ULL << xasm_singleParser::T__32)
      | (1ULL << xasm_singleParser::T__33)
      | (1ULL << xasm_singleParser::T__34)
      | (1ULL << xasm_singleParser::T__35)
      | (1ULL << xasm_singleParser::T__36))) != 0))) {
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

//----------------- IdContext ------------------------------------------------------------------

xasm_singleParser::IdContext::IdContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* xasm_singleParser::IdContext::ID() {
  return getToken(xasm_singleParser::ID, 0);
}


size_t xasm_singleParser::IdContext::getRuleIndex() const {
  return xasm_singleParser::RuleId;
}

void xasm_singleParser::IdContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterId(this);
}

void xasm_singleParser::IdContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitId(this);
}


antlrcpp::Any xasm_singleParser::IdContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitId(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::IdContext* xasm_singleParser::id() {
  IdContext *_localctx = _tracker.createInstance<IdContext>(_ctx, getState());
  enterRule(_localctx, 20, xasm_singleParser::RuleId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(206);
    match(xasm_singleParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RealContext ------------------------------------------------------------------

xasm_singleParser::RealContext::RealContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* xasm_singleParser::RealContext::REAL() {
  return getToken(xasm_singleParser::REAL, 0);
}


size_t xasm_singleParser::RealContext::getRuleIndex() const {
  return xasm_singleParser::RuleReal;
}

void xasm_singleParser::RealContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReal(this);
}

void xasm_singleParser::RealContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReal(this);
}


antlrcpp::Any xasm_singleParser::RealContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitReal(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::RealContext* xasm_singleParser::real() {
  RealContext *_localctx = _tracker.createInstance<RealContext>(_ctx, getState());
  enterRule(_localctx, 22, xasm_singleParser::RuleReal);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(208);
    match(xasm_singleParser::REAL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StringContext ------------------------------------------------------------------

xasm_singleParser::StringContext::StringContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* xasm_singleParser::StringContext::STRING() {
  return getToken(xasm_singleParser::STRING, 0);
}


size_t xasm_singleParser::StringContext::getRuleIndex() const {
  return xasm_singleParser::RuleString;
}

void xasm_singleParser::StringContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterString(this);
}

void xasm_singleParser::StringContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<xasm_singleListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitString(this);
}


antlrcpp::Any xasm_singleParser::StringContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<xasm_singleVisitor*>(visitor))
    return parserVisitor->visitString(this);
  else
    return visitor->visitChildren(this);
}

xasm_singleParser::StringContext* xasm_singleParser::string() {
  StringContext *_localctx = _tracker.createInstance<StringContext>(_ctx, getState());
  enterRule(_localctx, 24, xasm_singleParser::RuleString);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(210);
    match(xasm_singleParser::STRING);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool xasm_singleParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 8: return expSempred(dynamic_cast<ExpContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool xasm_singleParser::expSempred(ExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 19);
    case 1: return precpred(_ctx, 18);
    case 2: return precpred(_ctx, 17);
    case 3: return precpred(_ctx, 16);
    case 4: return precpred(_ctx, 15);
    case 5: return precpred(_ctx, 14);
    case 6: return precpred(_ctx, 8);
    case 7: return precpred(_ctx, 13);
    case 8: return precpred(_ctx, 12);
    case 9: return precpred(_ctx, 11);
    case 10: return precpred(_ctx, 10);
    case 11: return precpred(_ctx, 5);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> xasm_singleParser::_decisionToDFA;
atn::PredictionContextCache xasm_singleParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN xasm_singleParser::_atn;
std::vector<uint16_t> xasm_singleParser::_serializedATN;

std::vector<std::string> xasm_singleParser::_ruleNames = {
  "line", "statement", "comment", "qinst", "cinst", "cpp_type", "compare", 
  "explist", "exp", "unaryop", "id", "real", "string"
};

std::vector<std::string> xasm_singleParser::_literalNames = {
  "", "'('", "')'", "';'", "'const'", "'='", "'++'", "'--'", "'for'", "'{'", 
  "'}'", "'=='", "'?'", "':'", "'auto'", "'&'", "'*'", "'>'", "'<'", "'>='", 
  "'<='", "','", "'+'", "'-'", "'/'", "'::'", "'<<'", "'.'", "'^'", "'['", 
  "']'", "'pi'", "'sin'", "'cos'", "'tan'", "'exp'", "'ln'", "'sqrt'"
};

std::vector<std::string> xasm_singleParser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "COMMENT", "ID", "REAL", "INT", "STRING", "WS", "EOL"
};

dfa::Vocabulary xasm_singleParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> xasm_singleParser::_tokenNames;

xasm_singleParser::Initializer::Initializer() {
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

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x2e, 0xd7, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 0x9, 
    0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 0x4, 
    0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 0x9, 
    0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 0x3, 
    0x2, 0x3, 0x2, 0x5, 0x2, 0x1f, 0xa, 0x2, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 
    0x23, 0xa, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x5, 0x6, 0x2e, 0xa, 0x6, 0x3, 0x6, 
    0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x34, 0xa, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x4b, 
    0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
    0x5, 0x6, 0x53, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x57, 0xa, 0x6, 
    0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
    0x3, 0x6, 0x5, 0x6, 0x61, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x71, 0xa, 0x6, 0x3, 0x7, 
    0x3, 0x7, 0x5, 0x7, 0x75, 0xa, 0x7, 0x3, 0x7, 0x5, 0x7, 0x78, 0xa, 0x7, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x7, 0x9, 0x7f, 0xa, 
    0x9, 0xc, 0x9, 0xe, 0x9, 0x82, 0xb, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
    0x5, 0xa, 0x95, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x7, 0xa, 0xca, 0xa, 0xa, 
    0xc, 0xa, 0xe, 0xa, 0xcd, 0xb, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x2, 0x3, 0x12, 
    0xf, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 
    0x1a, 0x2, 0x6, 0x3, 0x2, 0x8, 0x9, 0x3, 0x2, 0x11, 0x12, 0x3, 0x2, 
    0x13, 0x16, 0x3, 0x2, 0x22, 0x27, 0x2, 0xee, 0x2, 0x1e, 0x3, 0x2, 0x2, 
    0x2, 0x4, 0x22, 0x3, 0x2, 0x2, 0x2, 0x6, 0x24, 0x3, 0x2, 0x2, 0x2, 0x8, 
    0x26, 0x3, 0x2, 0x2, 0x2, 0xa, 0x70, 0x3, 0x2, 0x2, 0x2, 0xc, 0x77, 
    0x3, 0x2, 0x2, 0x2, 0xe, 0x79, 0x3, 0x2, 0x2, 0x2, 0x10, 0x7b, 0x3, 
    0x2, 0x2, 0x2, 0x12, 0x94, 0x3, 0x2, 0x2, 0x2, 0x14, 0xce, 0x3, 0x2, 
    0x2, 0x2, 0x16, 0xd0, 0x3, 0x2, 0x2, 0x2, 0x18, 0xd2, 0x3, 0x2, 0x2, 
    0x2, 0x1a, 0xd4, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x1f, 0x5, 0x4, 0x3, 0x2, 
    0x1d, 0x1f, 0x5, 0x6, 0x4, 0x2, 0x1e, 0x1c, 0x3, 0x2, 0x2, 0x2, 0x1e, 
    0x1d, 0x3, 0x2, 0x2, 0x2, 0x1f, 0x3, 0x3, 0x2, 0x2, 0x2, 0x20, 0x23, 
    0x5, 0x8, 0x5, 0x2, 0x21, 0x23, 0x5, 0xa, 0x6, 0x2, 0x22, 0x20, 0x3, 
    0x2, 0x2, 0x2, 0x22, 0x21, 0x3, 0x2, 0x2, 0x2, 0x23, 0x5, 0x3, 0x2, 
    0x2, 0x2, 0x24, 0x25, 0x7, 0x28, 0x2, 0x2, 0x25, 0x7, 0x3, 0x2, 0x2, 
    0x2, 0x26, 0x27, 0x5, 0x16, 0xc, 0x2, 0x27, 0x28, 0x7, 0x3, 0x2, 0x2, 
    0x28, 0x29, 0x5, 0x10, 0x9, 0x2, 0x29, 0x2a, 0x7, 0x4, 0x2, 0x2, 0x2a, 
    0x2b, 0x7, 0x5, 0x2, 0x2, 0x2b, 0x9, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x2e, 
    0x7, 0x6, 0x2, 0x2, 0x2d, 0x2c, 0x3, 0x2, 0x2, 0x2, 0x2d, 0x2e, 0x3, 
    0x2, 0x2, 0x2, 0x2e, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x2f, 0x30, 0x5, 0xc, 
    0x7, 0x2, 0x30, 0x33, 0x5, 0x12, 0xa, 0x2, 0x31, 0x32, 0x7, 0x7, 0x2, 
    0x2, 0x32, 0x34, 0x5, 0x12, 0xa, 0x2, 0x33, 0x31, 0x3, 0x2, 0x2, 0x2, 
    0x33, 0x34, 0x3, 0x2, 0x2, 0x2, 0x34, 0x35, 0x3, 0x2, 0x2, 0x2, 0x35, 
    0x36, 0x7, 0x5, 0x2, 0x2, 0x36, 0x71, 0x3, 0x2, 0x2, 0x2, 0x37, 0x38, 
    0x5, 0x12, 0xa, 0x2, 0x38, 0x39, 0x7, 0x8, 0x2, 0x2, 0x39, 0x3a, 0x7, 
    0x5, 0x2, 0x2, 0x3a, 0x71, 0x3, 0x2, 0x2, 0x2, 0x3b, 0x3c, 0x5, 0x12, 
    0xa, 0x2, 0x3c, 0x3d, 0x7, 0x9, 0x2, 0x2, 0x3d, 0x3e, 0x7, 0x5, 0x2, 
    0x2, 0x3e, 0x71, 0x3, 0x2, 0x2, 0x2, 0x3f, 0x40, 0x7, 0xa, 0x2, 0x2, 
    0x40, 0x41, 0x7, 0x3, 0x2, 0x2, 0x41, 0x42, 0x5, 0xc, 0x7, 0x2, 0x42, 
    0x43, 0x5, 0x12, 0xa, 0x2, 0x43, 0x44, 0x7, 0x7, 0x2, 0x2, 0x44, 0x45, 
    0x5, 0x12, 0xa, 0x2, 0x45, 0x4a, 0x7, 0x5, 0x2, 0x2, 0x46, 0x47, 0x5, 
    0x12, 0xa, 0x2, 0x47, 0x48, 0x5, 0xe, 0x8, 0x2, 0x48, 0x49, 0x5, 0x12, 
    0xa, 0x2, 0x49, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x46, 0x3, 0x2, 0x2, 
    0x2, 0x4a, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x4b, 0x4c, 0x3, 0x2, 0x2, 0x2, 
    0x4c, 0x52, 0x7, 0x5, 0x2, 0x2, 0x4d, 0x4e, 0x5, 0x12, 0xa, 0x2, 0x4e, 
    0x4f, 0x9, 0x2, 0x2, 0x2, 0x4f, 0x53, 0x3, 0x2, 0x2, 0x2, 0x50, 0x51, 
    0x9, 0x2, 0x2, 0x2, 0x51, 0x53, 0x5, 0x12, 0xa, 0x2, 0x52, 0x4d, 0x3, 
    0x2, 0x2, 0x2, 0x52, 0x50, 0x3, 0x2, 0x2, 0x2, 0x52, 0x53, 0x3, 0x2, 
    0x2, 0x2, 0x53, 0x54, 0x3, 0x2, 0x2, 0x2, 0x54, 0x56, 0x7, 0x4, 0x2, 
    0x2, 0x55, 0x57, 0x7, 0xb, 0x2, 0x2, 0x56, 0x55, 0x3, 0x2, 0x2, 0x2, 
    0x56, 0x57, 0x3, 0x2, 0x2, 0x2, 0x57, 0x71, 0x3, 0x2, 0x2, 0x2, 0x58, 
    0x71, 0x7, 0xc, 0x2, 0x2, 0x59, 0x5a, 0x5, 0x12, 0xa, 0x2, 0x5a, 0x5b, 
    0x7, 0x3, 0x2, 0x2, 0x5b, 0x5c, 0x5, 0x10, 0x9, 0x2, 0x5c, 0x5d, 0x7, 
    0x4, 0x2, 0x2, 0x5d, 0x5e, 0x7, 0x5, 0x2, 0x2, 0x5e, 0x71, 0x3, 0x2, 
    0x2, 0x2, 0x5f, 0x61, 0x7, 0x6, 0x2, 0x2, 0x60, 0x5f, 0x3, 0x2, 0x2, 
    0x2, 0x60, 0x61, 0x3, 0x2, 0x2, 0x2, 0x61, 0x62, 0x3, 0x2, 0x2, 0x2, 
    0x62, 0x63, 0x5, 0xc, 0x7, 0x2, 0x63, 0x64, 0x5, 0x12, 0xa, 0x2, 0x64, 
    0x65, 0x7, 0x7, 0x2, 0x2, 0x65, 0x66, 0x7, 0x3, 0x2, 0x2, 0x66, 0x67, 
    0x5, 0x12, 0xa, 0x2, 0x67, 0x68, 0x7, 0xd, 0x2, 0x2, 0x68, 0x69, 0x5, 
    0x12, 0xa, 0x2, 0x69, 0x6a, 0x7, 0x4, 0x2, 0x2, 0x6a, 0x6b, 0x7, 0xe, 
    0x2, 0x2, 0x6b, 0x6c, 0x5, 0x12, 0xa, 0x2, 0x6c, 0x6d, 0x7, 0xf, 0x2, 
    0x2, 0x6d, 0x6e, 0x5, 0x12, 0xa, 0x2, 0x6e, 0x6f, 0x7, 0x5, 0x2, 0x2, 
    0x6f, 0x71, 0x3, 0x2, 0x2, 0x2, 0x70, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x70, 
    0x37, 0x3, 0x2, 0x2, 0x2, 0x70, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x70, 0x3f, 
    0x3, 0x2, 0x2, 0x2, 0x70, 0x58, 0x3, 0x2, 0x2, 0x2, 0x70, 0x59, 0x3, 
    0x2, 0x2, 0x2, 0x70, 0x60, 0x3, 0x2, 0x2, 0x2, 0x71, 0xb, 0x3, 0x2, 
    0x2, 0x2, 0x72, 0x74, 0x7, 0x10, 0x2, 0x2, 0x73, 0x75, 0x9, 0x3, 0x2, 
    0x2, 0x74, 0x73, 0x3, 0x2, 0x2, 0x2, 0x74, 0x75, 0x3, 0x2, 0x2, 0x2, 
    0x75, 0x78, 0x3, 0x2, 0x2, 0x2, 0x76, 0x78, 0x5, 0x12, 0xa, 0x2, 0x77, 
    0x72, 0x3, 0x2, 0x2, 0x2, 0x77, 0x76, 0x3, 0x2, 0x2, 0x2, 0x78, 0xd, 
    0x3, 0x2, 0x2, 0x2, 0x79, 0x7a, 0x9, 0x4, 0x2, 0x2, 0x7a, 0xf, 0x3, 
    0x2, 0x2, 0x2, 0x7b, 0x80, 0x5, 0x12, 0xa, 0x2, 0x7c, 0x7d, 0x7, 0x17, 
    0x2, 0x2, 0x7d, 0x7f, 0x5, 0x12, 0xa, 0x2, 0x7e, 0x7c, 0x3, 0x2, 0x2, 
    0x2, 0x7f, 0x82, 0x3, 0x2, 0x2, 0x2, 0x80, 0x7e, 0x3, 0x2, 0x2, 0x2, 
    0x80, 0x81, 0x3, 0x2, 0x2, 0x2, 0x81, 0x11, 0x3, 0x2, 0x2, 0x2, 0x82, 
    0x80, 0x3, 0x2, 0x2, 0x2, 0x83, 0x84, 0x8, 0xa, 0x1, 0x2, 0x84, 0x95, 
    0x5, 0x16, 0xc, 0x2, 0x85, 0x86, 0x7, 0x19, 0x2, 0x2, 0x86, 0x95, 0x5, 
    0x12, 0xa, 0xb, 0x87, 0x88, 0x7, 0x3, 0x2, 0x2, 0x88, 0x89, 0x5, 0x12, 
    0xa, 0x2, 0x89, 0x8a, 0x7, 0x4, 0x2, 0x2, 0x8a, 0x95, 0x3, 0x2, 0x2, 
    0x2, 0x8b, 0x8c, 0x5, 0x14, 0xb, 0x2, 0x8c, 0x8d, 0x7, 0x3, 0x2, 0x2, 
    0x8d, 0x8e, 0x5, 0x12, 0xa, 0x2, 0x8e, 0x8f, 0x7, 0x4, 0x2, 0x2, 0x8f, 
    0x95, 0x3, 0x2, 0x2, 0x2, 0x90, 0x95, 0x5, 0x1a, 0xe, 0x2, 0x91, 0x95, 
    0x5, 0x18, 0xd, 0x2, 0x92, 0x95, 0x7, 0x2b, 0x2, 0x2, 0x93, 0x95, 0x7, 
    0x21, 0x2, 0x2, 0x94, 0x83, 0x3, 0x2, 0x2, 0x2, 0x94, 0x85, 0x3, 0x2, 
    0x2, 0x2, 0x94, 0x87, 0x3, 0x2, 0x2, 0x2, 0x94, 0x8b, 0x3, 0x2, 0x2, 
    0x2, 0x94, 0x90, 0x3, 0x2, 0x2, 0x2, 0x94, 0x91, 0x3, 0x2, 0x2, 0x2, 
    0x94, 0x92, 0x3, 0x2, 0x2, 0x2, 0x94, 0x93, 0x3, 0x2, 0x2, 0x2, 0x95, 
    0xcb, 0x3, 0x2, 0x2, 0x2, 0x96, 0x97, 0xc, 0x15, 0x2, 0x2, 0x97, 0x98, 
    0x7, 0x18, 0x2, 0x2, 0x98, 0xca, 0x5, 0x12, 0xa, 0x16, 0x99, 0x9a, 0xc, 
    0x14, 0x2, 0x2, 0x9a, 0x9b, 0x7, 0x19, 0x2, 0x2, 0x9b, 0xca, 0x5, 0x12, 
    0xa, 0x15, 0x9c, 0x9d, 0xc, 0x13, 0x2, 0x2, 0x9d, 0x9e, 0x7, 0x12, 0x2, 
    0x2, 0x9e, 0xca, 0x5, 0x12, 0xa, 0x14, 0x9f, 0xa0, 0xc, 0x12, 0x2, 0x2, 
    0xa0, 0xa1, 0x7, 0x1a, 0x2, 0x2, 0xa1, 0xca, 0x5, 0x12, 0xa, 0x13, 0xa2, 
    0xa3, 0xc, 0x11, 0x2, 0x2, 0xa3, 0xa4, 0x7, 0x1b, 0x2, 0x2, 0xa4, 0xca, 
    0x5, 0x12, 0xa, 0x12, 0xa5, 0xa6, 0xc, 0x10, 0x2, 0x2, 0xa6, 0xa7, 0x7, 
    0x1c, 0x2, 0x2, 0xa7, 0xca, 0x5, 0x12, 0xa, 0x11, 0xa8, 0xa9, 0xc, 0xa, 
    0x2, 0x2, 0xa9, 0xaa, 0x7, 0x1e, 0x2, 0x2, 0xaa, 0xca, 0x5, 0x12, 0xa, 
    0xb, 0xab, 0xac, 0xc, 0xf, 0x2, 0x2, 0xac, 0xad, 0x7, 0x14, 0x2, 0x2, 
    0xad, 0xae, 0x5, 0x12, 0xa, 0x2, 0xae, 0xaf, 0x7, 0x13, 0x2, 0x2, 0xaf, 
    0xca, 0x3, 0x2, 0x2, 0x2, 0xb0, 0xb1, 0xc, 0xe, 0x2, 0x2, 0xb1, 0xb2, 
    0x7, 0x1b, 0x2, 0x2, 0xb2, 0xb3, 0x5, 0x12, 0xa, 0x2, 0xb3, 0xb4, 0x7, 
    0x3, 0x2, 0x2, 0xb4, 0xb5, 0x5, 0x10, 0x9, 0x2, 0xb5, 0xb6, 0x7, 0x4, 
    0x2, 0x2, 0xb6, 0xca, 0x3, 0x2, 0x2, 0x2, 0xb7, 0xb8, 0xc, 0xd, 0x2, 
    0x2, 0xb8, 0xb9, 0x7, 0x1d, 0x2, 0x2, 0xb9, 0xba, 0x5, 0x12, 0xa, 0x2, 
    0xba, 0xbb, 0x7, 0x3, 0x2, 0x2, 0xbb, 0xbc, 0x7, 0x4, 0x2, 0x2, 0xbc, 
    0xca, 0x3, 0x2, 0x2, 0x2, 0xbd, 0xbe, 0xc, 0xc, 0x2, 0x2, 0xbe, 0xbf, 
    0x7, 0x1d, 0x2, 0x2, 0xbf, 0xc0, 0x5, 0x12, 0xa, 0x2, 0xc0, 0xc1, 0x7, 
    0x3, 0x2, 0x2, 0xc1, 0xc2, 0x5, 0x10, 0x9, 0x2, 0xc2, 0xc3, 0x7, 0x4, 
    0x2, 0x2, 0xc3, 0xca, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc5, 0xc, 0x7, 0x2, 
    0x2, 0xc5, 0xc6, 0x7, 0x1f, 0x2, 0x2, 0xc6, 0xc7, 0x5, 0x12, 0xa, 0x2, 
    0xc7, 0xc8, 0x7, 0x20, 0x2, 0x2, 0xc8, 0xca, 0x3, 0x2, 0x2, 0x2, 0xc9, 
    0x96, 0x3, 0x2, 0x2, 0x2, 0xc9, 0x99, 0x3, 0x2, 0x2, 0x2, 0xc9, 0x9c, 
    0x3, 0x2, 0x2, 0x2, 0xc9, 0x9f, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xa2, 0x3, 
    0x2, 0x2, 0x2, 0xc9, 0xa5, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xa8, 0x3, 0x2, 
    0x2, 0x2, 0xc9, 0xab, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xb0, 0x3, 0x2, 0x2, 
    0x2, 0xc9, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xbd, 0x3, 0x2, 0x2, 0x2, 
    0xc9, 0xc4, 0x3, 0x2, 0x2, 0x2, 0xca, 0xcd, 0x3, 0x2, 0x2, 0x2, 0xcb, 
    0xc9, 0x3, 0x2, 0x2, 0x2, 0xcb, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xcc, 0x13, 
    0x3, 0x2, 0x2, 0x2, 0xcd, 0xcb, 0x3, 0x2, 0x2, 0x2, 0xce, 0xcf, 0x9, 
    0x5, 0x2, 0x2, 0xcf, 0x15, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xd1, 0x7, 0x29, 
    0x2, 0x2, 0xd1, 0x17, 0x3, 0x2, 0x2, 0x2, 0xd2, 0xd3, 0x7, 0x2a, 0x2, 
    0x2, 0xd3, 0x19, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd5, 0x7, 0x2c, 0x2, 0x2, 
    0xd5, 0x1b, 0x3, 0x2, 0x2, 0x2, 0x11, 0x1e, 0x22, 0x2d, 0x33, 0x4a, 
    0x52, 0x56, 0x60, 0x70, 0x74, 0x77, 0x80, 0x94, 0xc9, 0xcb, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

xasm_singleParser::Initializer xasm_singleParser::_init;

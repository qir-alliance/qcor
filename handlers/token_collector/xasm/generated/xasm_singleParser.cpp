
// Generated from xasm_single.g4 by ANTLR 4.9.2


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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
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
      case xasm_singleParser::T__10:
      case xasm_singleParser::T__11:
      case xasm_singleParser::T__15:
      case xasm_singleParser::T__16:
      case xasm_singleParser::T__17:
      case xasm_singleParser::T__26:
      case xasm_singleParser::T__34:
      case xasm_singleParser::T__37:
      case xasm_singleParser::T__38:
      case xasm_singleParser::T__39:
      case xasm_singleParser::T__40:
      case xasm_singleParser::T__41:
      case xasm_singleParser::T__42:
      case xasm_singleParser::T__43:
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(135);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
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
        | (1ULL << xasm_singleParser::T__26)
        | (1ULL << xasm_singleParser::T__34)
        | (1ULL << xasm_singleParser::T__37)
        | (1ULL << xasm_singleParser::T__38)
        | (1ULL << xasm_singleParser::T__39)
        | (1ULL << xasm_singleParser::T__40)
        | (1ULL << xasm_singleParser::T__41)
        | (1ULL << xasm_singleParser::T__42)
        | (1ULL << xasm_singleParser::T__43)
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
        case xasm_singleParser::T__26:
        case xasm_singleParser::T__34:
        case xasm_singleParser::T__37:
        case xasm_singleParser::T__38:
        case xasm_singleParser::T__39:
        case xasm_singleParser::T__40:
        case xasm_singleParser::T__41:
        case xasm_singleParser::T__42:
        case xasm_singleParser::T__43:
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
      setState(90);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << xasm_singleParser::T__0)
        | (1ULL << xasm_singleParser::T__26)
        | (1ULL << xasm_singleParser::T__34)
        | (1ULL << xasm_singleParser::T__37)
        | (1ULL << xasm_singleParser::T__38)
        | (1ULL << xasm_singleParser::T__39)
        | (1ULL << xasm_singleParser::T__40)
        | (1ULL << xasm_singleParser::T__41)
        | (1ULL << xasm_singleParser::T__42)
        | (1ULL << xasm_singleParser::T__43)
        | (1ULL << xasm_singleParser::ID)
        | (1ULL << xasm_singleParser::REAL)
        | (1ULL << xasm_singleParser::INT)
        | (1ULL << xasm_singleParser::STRING))) != 0)) {
        setState(89);
        explist();
      }
      setState(92);
      match(xasm_singleParser::T__1);
      setState(93);
      match(xasm_singleParser::T__2);
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(95);
      match(xasm_singleParser::T__10);
      setState(96);
      match(xasm_singleParser::T__0);
      setState(97);
      explist();
      setState(98);
      match(xasm_singleParser::T__1);
      setState(100);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == xasm_singleParser::T__8) {
        setState(99);
        match(xasm_singleParser::T__8);
      }
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(102);
      match(xasm_singleParser::T__11);
      setState(103);
      match(xasm_singleParser::T__0);
      setState(104);
      explist();
      setState(105);
      match(xasm_singleParser::T__1);
      setState(107);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == xasm_singleParser::T__8) {
        setState(106);
        match(xasm_singleParser::T__8);
      }
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(110);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == xasm_singleParser::T__3) {
        setState(109);
        match(xasm_singleParser::T__3);
      }
      setState(112);
      dynamic_cast<CinstContext *>(_localctx)->type_name = cpp_type();
      setState(113);
      dynamic_cast<CinstContext *>(_localctx)->var_name = exp(0);
      setState(114);
      match(xasm_singleParser::T__4);
      setState(115);
      match(xasm_singleParser::T__0);
      setState(116);
      exp(0);
      setState(117);
      match(xasm_singleParser::T__12);
      setState(118);
      exp(0);
      setState(119);
      match(xasm_singleParser::T__1);
      setState(120);
      match(xasm_singleParser::T__13);
      setState(121);
      exp(0);
      setState(122);
      match(xasm_singleParser::T__14);
      setState(123);
      exp(0);
      setState(124);
      match(xasm_singleParser::T__2);
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(126);
      match(xasm_singleParser::T__15);
      setState(127);
      match(xasm_singleParser::T__2);
      break;
    }

    case 11: {
      enterOuterAlt(_localctx, 11);
      setState(128);
      match(xasm_singleParser::T__16);
      setState(129);
      match(xasm_singleParser::T__2);
      break;
    }

    case 12: {
      enterOuterAlt(_localctx, 12);
      setState(130);
      exp(0);
      setState(131);
      match(xasm_singleParser::T__4);
      setState(132);
      exp(0);
      setState(133);
      match(xasm_singleParser::T__2);
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(142);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case xasm_singleParser::T__17: {
        enterOuterAlt(_localctx, 1);
        setState(137);
        match(xasm_singleParser::T__17);
        setState(139);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == xasm_singleParser::T__18

        || _la == xasm_singleParser::T__19) {
          setState(138);
          _la = _input->LA(1);
          if (!(_la == xasm_singleParser::T__18

          || _la == xasm_singleParser::T__19)) {
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
      case xasm_singleParser::T__26:
      case xasm_singleParser::T__34:
      case xasm_singleParser::T__37:
      case xasm_singleParser::T__38:
      case xasm_singleParser::T__39:
      case xasm_singleParser::T__40:
      case xasm_singleParser::T__41:
      case xasm_singleParser::T__42:
      case xasm_singleParser::T__43:
      case xasm_singleParser::ID:
      case xasm_singleParser::REAL:
      case xasm_singleParser::INT:
      case xasm_singleParser::STRING: {
        enterOuterAlt(_localctx, 2);
        setState(141);
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(144);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << xasm_singleParser::T__20)
      | (1ULL << xasm_singleParser::T__21)
      | (1ULL << xasm_singleParser::T__22)
      | (1ULL << xasm_singleParser::T__23))) != 0))) {
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(146);
    exp(0);
    setState(151);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == xasm_singleParser::T__24) {
      setState(147);
      match(xasm_singleParser::T__24);
      setState(148);
      exp(0);
      setState(153);
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
    setState(173);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case xasm_singleParser::ID: {
        setState(155);
        id();
        break;
      }

      case xasm_singleParser::T__26: {
        setState(156);
        match(xasm_singleParser::T__26);
        setState(157);
        exp(13);
        break;
      }

      case xasm_singleParser::T__0: {
        setState(158);
        match(xasm_singleParser::T__0);
        setState(159);
        exp(0);
        setState(160);
        match(xasm_singleParser::T__1);
        break;
      }

      case xasm_singleParser::T__38:
      case xasm_singleParser::T__39:
      case xasm_singleParser::T__40:
      case xasm_singleParser::T__41:
      case xasm_singleParser::T__42:
      case xasm_singleParser::T__43: {
        setState(162);
        unaryop();
        setState(163);
        match(xasm_singleParser::T__0);
        setState(164);
        exp(0);
        setState(165);
        match(xasm_singleParser::T__1);
        break;
      }

      case xasm_singleParser::STRING: {
        setState(167);
        string();
        break;
      }

      case xasm_singleParser::REAL: {
        setState(168);
        real();
        break;
      }

      case xasm_singleParser::INT: {
        setState(169);
        match(xasm_singleParser::INT);
        break;
      }

      case xasm_singleParser::T__34: {
        setState(170);
        match(xasm_singleParser::T__34);
        break;
      }

      case xasm_singleParser::T__37: {
        setState(171);
        match(xasm_singleParser::T__37);
        setState(172);
        exp(1);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(240);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(238);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(175);

          if (!(precpred(_ctx, 23))) throw FailedPredicateException(this, "precpred(_ctx, 23)");
          setState(176);
          match(xasm_singleParser::T__25);
          setState(177);
          exp(24);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(178);

          if (!(precpred(_ctx, 22))) throw FailedPredicateException(this, "precpred(_ctx, 22)");
          setState(179);
          match(xasm_singleParser::T__26);
          setState(180);
          exp(23);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(181);

          if (!(precpred(_ctx, 21))) throw FailedPredicateException(this, "precpred(_ctx, 21)");
          setState(182);
          match(xasm_singleParser::T__19);
          setState(183);
          exp(22);
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(184);

          if (!(precpred(_ctx, 20))) throw FailedPredicateException(this, "precpred(_ctx, 20)");
          setState(185);
          match(xasm_singleParser::T__27);
          setState(186);
          exp(21);
          break;
        }

        case 5: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(187);

          if (!(precpred(_ctx, 19))) throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(188);
          match(xasm_singleParser::T__28);
          setState(189);
          exp(20);
          break;
        }

        case 6: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(190);

          if (!(precpred(_ctx, 18))) throw FailedPredicateException(this, "precpred(_ctx, 18)");
          setState(191);
          match(xasm_singleParser::T__29);
          setState(192);
          exp(19);
          break;
        }

        case 7: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(193);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(194);
          match(xasm_singleParser::T__31);
          setState(195);
          exp(13);
          break;
        }

        case 8: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(196);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(197);
          match(xasm_singleParser::T__35);
          setState(198);
          exp(4);
          break;
        }

        case 9: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(199);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(200);
          match(xasm_singleParser::T__36);
          setState(201);
          exp(3);
          break;
        }

        case 10: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(202);

          if (!(precpred(_ctx, 17))) throw FailedPredicateException(this, "precpred(_ctx, 17)");
          setState(203);
          match(xasm_singleParser::T__21);
          setState(204);
          exp(0);
          setState(205);
          match(xasm_singleParser::T__20);
          break;
        }

        case 11: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(207);

          if (!(precpred(_ctx, 16))) throw FailedPredicateException(this, "precpred(_ctx, 16)");
          setState(208);
          match(xasm_singleParser::T__28);
          setState(209);
          exp(0);
          setState(210);
          match(xasm_singleParser::T__0);
          setState(211);
          explist();
          setState(212);
          match(xasm_singleParser::T__1);
          break;
        }

        case 12: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(214);

          if (!(precpred(_ctx, 15))) throw FailedPredicateException(this, "precpred(_ctx, 15)");
          setState(215);
          match(xasm_singleParser::T__30);
          setState(216);
          exp(0);
          setState(217);
          match(xasm_singleParser::T__0);
          setState(218);
          match(xasm_singleParser::T__1);
          break;
        }

        case 13: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(220);

          if (!(precpred(_ctx, 14))) throw FailedPredicateException(this, "precpred(_ctx, 14)");
          setState(221);
          match(xasm_singleParser::T__30);
          setState(222);
          exp(0);
          setState(223);
          match(xasm_singleParser::T__0);
          setState(224);
          explist();
          setState(225);
          match(xasm_singleParser::T__1);
          break;
        }

        case 14: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(227);

          if (!(precpred(_ctx, 9))) throw FailedPredicateException(this, "precpred(_ctx, 9)");
          setState(228);
          match(xasm_singleParser::T__0);
          setState(230);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if ((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & ((1ULL << xasm_singleParser::T__0)
            | (1ULL << xasm_singleParser::T__26)
            | (1ULL << xasm_singleParser::T__34)
            | (1ULL << xasm_singleParser::T__37)
            | (1ULL << xasm_singleParser::T__38)
            | (1ULL << xasm_singleParser::T__39)
            | (1ULL << xasm_singleParser::T__40)
            | (1ULL << xasm_singleParser::T__41)
            | (1ULL << xasm_singleParser::T__42)
            | (1ULL << xasm_singleParser::T__43)
            | (1ULL << xasm_singleParser::ID)
            | (1ULL << xasm_singleParser::REAL)
            | (1ULL << xasm_singleParser::INT)
            | (1ULL << xasm_singleParser::STRING))) != 0)) {
            setState(229);
            explist();
          }
          setState(232);
          match(xasm_singleParser::T__1);
          break;
        }

        case 15: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(233);

          if (!(precpred(_ctx, 8))) throw FailedPredicateException(this, "precpred(_ctx, 8)");
          setState(234);
          match(xasm_singleParser::T__32);
          setState(235);
          exp(0);
          setState(236);
          match(xasm_singleParser::T__33);
          break;
        }

        default:
          break;
        } 
      }
      setState(242);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx);
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(243);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << xasm_singleParser::T__38)
      | (1ULL << xasm_singleParser::T__39)
      | (1ULL << xasm_singleParser::T__40)
      | (1ULL << xasm_singleParser::T__41)
      | (1ULL << xasm_singleParser::T__42)
      | (1ULL << xasm_singleParser::T__43))) != 0))) {
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(245);
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(247);
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

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(249);
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
    case 0: return precpred(_ctx, 23);
    case 1: return precpred(_ctx, 22);
    case 2: return precpred(_ctx, 21);
    case 3: return precpred(_ctx, 20);
    case 4: return precpred(_ctx, 19);
    case 5: return precpred(_ctx, 18);
    case 6: return precpred(_ctx, 12);
    case 7: return precpred(_ctx, 3);
    case 8: return precpred(_ctx, 2);
    case 9: return precpred(_ctx, 17);
    case 10: return precpred(_ctx, 16);
    case 11: return precpred(_ctx, 15);
    case 12: return precpred(_ctx, 14);
    case 13: return precpred(_ctx, 9);
    case 14: return precpred(_ctx, 8);

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
  "'}'", "'if'", "'else'", "'=='", "'\u003F'", "':'", "'break'", "'return'", 
  "'auto'", "'&'", "'*'", "'>'", "'<'", "'>='", "'<='", "','", "'+'", "'-'", 
  "'/'", "'::'", "'<<'", "'.'", "'^'", "'['", "']'", "'pi'", "'&&'", "'||'", 
  "'!'", "'sin'", "'cos'", "'tan'", "'exp'", "'ln'", "'sqrt'"
};

std::vector<std::string> xasm_singleParser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "COMMENT", "ID", "REAL", "INT", "STRING", 
  "WS", "EOL"
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

  static const uint16_t serializedATNSegment0[] = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
       0x3, 0x35, 0xfe, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
       0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 
       0x7, 0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 
       0x4, 0xb, 0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 
       0xe, 0x9, 0xe, 0x3, 0x2, 0x3, 0x2, 0x5, 0x2, 0x1f, 0xa, 0x2, 0x3, 
       0x3, 0x3, 0x3, 0x5, 0x3, 0x23, 0xa, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 
       0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 
       0x5, 0x6, 0x2e, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
       0x5, 0x6, 0x34, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
       0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x4b, 0xa, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 
       0x6, 0x53, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x57, 0xa, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x5d, 0xa, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
       0x6, 0x3, 0x6, 0x5, 0x6, 0x67, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
       0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x6e, 0xa, 0x6, 0x3, 0x6, 0x5, 
       0x6, 0x71, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
       0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
       0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
       0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 
       0x8a, 0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x8e, 0xa, 0x7, 0x3, 
       0x7, 0x5, 0x7, 0x91, 0xa, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 
       0x9, 0x3, 0x9, 0x7, 0x9, 0x98, 0xa, 0x9, 0xc, 0x9, 0xe, 0x9, 0x9b, 
       0xb, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
       0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
       0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
       0xa, 0x5, 0xa, 0xb0, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
       0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
       0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
       0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
       0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
       0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
       0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
       0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
       0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 
       0xa, 0xe9, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
       0xa, 0x3, 0xa, 0x7, 0xa, 0xf1, 0xa, 0xa, 0xc, 0xa, 0xe, 0xa, 0xf4, 
       0xb, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xd, 0x3, 
       0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x2, 0x3, 0x12, 0xf, 0x2, 0x4, 
       0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x2, 
       0x6, 0x3, 0x2, 0x8, 0x9, 0x3, 0x2, 0x15, 0x16, 0x3, 0x2, 0x17, 0x1a, 
       0x3, 0x2, 0x29, 0x2e, 0x2, 0x122, 0x2, 0x1e, 0x3, 0x2, 0x2, 0x2, 
       0x4, 0x22, 0x3, 0x2, 0x2, 0x2, 0x6, 0x24, 0x3, 0x2, 0x2, 0x2, 0x8, 
       0x26, 0x3, 0x2, 0x2, 0x2, 0xa, 0x89, 0x3, 0x2, 0x2, 0x2, 0xc, 0x90, 
       0x3, 0x2, 0x2, 0x2, 0xe, 0x92, 0x3, 0x2, 0x2, 0x2, 0x10, 0x94, 0x3, 
       0x2, 0x2, 0x2, 0x12, 0xaf, 0x3, 0x2, 0x2, 0x2, 0x14, 0xf5, 0x3, 0x2, 
       0x2, 0x2, 0x16, 0xf7, 0x3, 0x2, 0x2, 0x2, 0x18, 0xf9, 0x3, 0x2, 0x2, 
       0x2, 0x1a, 0xfb, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x1f, 0x5, 0x4, 0x3, 0x2, 
       0x1d, 0x1f, 0x5, 0x6, 0x4, 0x2, 0x1e, 0x1c, 0x3, 0x2, 0x2, 0x2, 0x1e, 
       0x1d, 0x3, 0x2, 0x2, 0x2, 0x1f, 0x3, 0x3, 0x2, 0x2, 0x2, 0x20, 0x23, 
       0x5, 0x8, 0x5, 0x2, 0x21, 0x23, 0x5, 0xa, 0x6, 0x2, 0x22, 0x20, 0x3, 
       0x2, 0x2, 0x2, 0x22, 0x21, 0x3, 0x2, 0x2, 0x2, 0x23, 0x5, 0x3, 0x2, 
       0x2, 0x2, 0x24, 0x25, 0x7, 0x2f, 0x2, 0x2, 0x25, 0x7, 0x3, 0x2, 0x2, 
       0x2, 0x26, 0x27, 0x5, 0x16, 0xc, 0x2, 0x27, 0x28, 0x7, 0x3, 0x2, 
       0x2, 0x28, 0x29, 0x5, 0x10, 0x9, 0x2, 0x29, 0x2a, 0x7, 0x4, 0x2, 
       0x2, 0x2a, 0x2b, 0x7, 0x5, 0x2, 0x2, 0x2b, 0x9, 0x3, 0x2, 0x2, 0x2, 
       0x2c, 0x2e, 0x7, 0x6, 0x2, 0x2, 0x2d, 0x2c, 0x3, 0x2, 0x2, 0x2, 0x2d, 
       0x2e, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x2f, 0x30, 
       0x5, 0xc, 0x7, 0x2, 0x30, 0x33, 0x5, 0x12, 0xa, 0x2, 0x31, 0x32, 
       0x7, 0x7, 0x2, 0x2, 0x32, 0x34, 0x5, 0x12, 0xa, 0x2, 0x33, 0x31, 
       0x3, 0x2, 0x2, 0x2, 0x33, 0x34, 0x3, 0x2, 0x2, 0x2, 0x34, 0x35, 0x3, 
       0x2, 0x2, 0x2, 0x35, 0x36, 0x7, 0x5, 0x2, 0x2, 0x36, 0x8a, 0x3, 0x2, 
       0x2, 0x2, 0x37, 0x38, 0x5, 0x12, 0xa, 0x2, 0x38, 0x39, 0x7, 0x8, 
       0x2, 0x2, 0x39, 0x3a, 0x7, 0x5, 0x2, 0x2, 0x3a, 0x8a, 0x3, 0x2, 0x2, 
       0x2, 0x3b, 0x3c, 0x5, 0x12, 0xa, 0x2, 0x3c, 0x3d, 0x7, 0x9, 0x2, 
       0x2, 0x3d, 0x3e, 0x7, 0x5, 0x2, 0x2, 0x3e, 0x8a, 0x3, 0x2, 0x2, 0x2, 
       0x3f, 0x40, 0x7, 0xa, 0x2, 0x2, 0x40, 0x41, 0x7, 0x3, 0x2, 0x2, 0x41, 
       0x42, 0x5, 0xc, 0x7, 0x2, 0x42, 0x43, 0x5, 0x12, 0xa, 0x2, 0x43, 
       0x44, 0x7, 0x7, 0x2, 0x2, 0x44, 0x45, 0x5, 0x12, 0xa, 0x2, 0x45, 
       0x4a, 0x7, 0x5, 0x2, 0x2, 0x46, 0x47, 0x5, 0x12, 0xa, 0x2, 0x47, 
       0x48, 0x5, 0xe, 0x8, 0x2, 0x48, 0x49, 0x5, 0x12, 0xa, 0x2, 0x49, 
       0x4b, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x46, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x4b, 
       0x3, 0x2, 0x2, 0x2, 0x4b, 0x4c, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x52, 0x7, 
       0x5, 0x2, 0x2, 0x4d, 0x4e, 0x5, 0x12, 0xa, 0x2, 0x4e, 0x4f, 0x9, 
       0x2, 0x2, 0x2, 0x4f, 0x53, 0x3, 0x2, 0x2, 0x2, 0x50, 0x51, 0x9, 0x2, 
       0x2, 0x2, 0x51, 0x53, 0x5, 0x12, 0xa, 0x2, 0x52, 0x4d, 0x3, 0x2, 
       0x2, 0x2, 0x52, 0x50, 0x3, 0x2, 0x2, 0x2, 0x52, 0x53, 0x3, 0x2, 0x2, 
       0x2, 0x53, 0x54, 0x3, 0x2, 0x2, 0x2, 0x54, 0x56, 0x7, 0x4, 0x2, 0x2, 
       0x55, 0x57, 0x7, 0xb, 0x2, 0x2, 0x56, 0x55, 0x3, 0x2, 0x2, 0x2, 0x56, 
       0x57, 0x3, 0x2, 0x2, 0x2, 0x57, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x58, 0x8a, 
       0x7, 0xc, 0x2, 0x2, 0x59, 0x5a, 0x5, 0x12, 0xa, 0x2, 0x5a, 0x5c, 
       0x7, 0x3, 0x2, 0x2, 0x5b, 0x5d, 0x5, 0x10, 0x9, 0x2, 0x5c, 0x5b, 
       0x3, 0x2, 0x2, 0x2, 0x5c, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x5d, 0x5e, 0x3, 
       0x2, 0x2, 0x2, 0x5e, 0x5f, 0x7, 0x4, 0x2, 0x2, 0x5f, 0x60, 0x7, 0x5, 
       0x2, 0x2, 0x60, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x61, 0x62, 0x7, 0xd, 0x2, 
       0x2, 0x62, 0x63, 0x7, 0x3, 0x2, 0x2, 0x63, 0x64, 0x5, 0x10, 0x9, 
       0x2, 0x64, 0x66, 0x7, 0x4, 0x2, 0x2, 0x65, 0x67, 0x7, 0xb, 0x2, 0x2, 
       0x66, 0x65, 0x3, 0x2, 0x2, 0x2, 0x66, 0x67, 0x3, 0x2, 0x2, 0x2, 0x67, 
       0x8a, 0x3, 0x2, 0x2, 0x2, 0x68, 0x69, 0x7, 0xe, 0x2, 0x2, 0x69, 0x6a, 
       0x7, 0x3, 0x2, 0x2, 0x6a, 0x6b, 0x5, 0x10, 0x9, 0x2, 0x6b, 0x6d, 
       0x7, 0x4, 0x2, 0x2, 0x6c, 0x6e, 0x7, 0xb, 0x2, 0x2, 0x6d, 0x6c, 0x3, 
       0x2, 0x2, 0x2, 0x6d, 0x6e, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x8a, 0x3, 0x2, 
       0x2, 0x2, 0x6f, 0x71, 0x7, 0x6, 0x2, 0x2, 0x70, 0x6f, 0x3, 0x2, 0x2, 
       0x2, 0x70, 0x71, 0x3, 0x2, 0x2, 0x2, 0x71, 0x72, 0x3, 0x2, 0x2, 0x2, 
       0x72, 0x73, 0x5, 0xc, 0x7, 0x2, 0x73, 0x74, 0x5, 0x12, 0xa, 0x2, 
       0x74, 0x75, 0x7, 0x7, 0x2, 0x2, 0x75, 0x76, 0x7, 0x3, 0x2, 0x2, 0x76, 
       0x77, 0x5, 0x12, 0xa, 0x2, 0x77, 0x78, 0x7, 0xf, 0x2, 0x2, 0x78, 
       0x79, 0x5, 0x12, 0xa, 0x2, 0x79, 0x7a, 0x7, 0x4, 0x2, 0x2, 0x7a, 
       0x7b, 0x7, 0x10, 0x2, 0x2, 0x7b, 0x7c, 0x5, 0x12, 0xa, 0x2, 0x7c, 
       0x7d, 0x7, 0x11, 0x2, 0x2, 0x7d, 0x7e, 0x5, 0x12, 0xa, 0x2, 0x7e, 
       0x7f, 0x7, 0x5, 0x2, 0x2, 0x7f, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x80, 0x81, 
       0x7, 0x12, 0x2, 0x2, 0x81, 0x8a, 0x7, 0x5, 0x2, 0x2, 0x82, 0x83, 
       0x7, 0x13, 0x2, 0x2, 0x83, 0x8a, 0x7, 0x5, 0x2, 0x2, 0x84, 0x85, 
       0x5, 0x12, 0xa, 0x2, 0x85, 0x86, 0x7, 0x7, 0x2, 0x2, 0x86, 0x87, 
       0x5, 0x12, 0xa, 0x2, 0x87, 0x88, 0x7, 0x5, 0x2, 0x2, 0x88, 0x8a, 
       0x3, 0x2, 0x2, 0x2, 0x89, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x89, 0x37, 0x3, 
       0x2, 0x2, 0x2, 0x89, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x89, 0x3f, 0x3, 0x2, 
       0x2, 0x2, 0x89, 0x58, 0x3, 0x2, 0x2, 0x2, 0x89, 0x59, 0x3, 0x2, 0x2, 
       0x2, 0x89, 0x61, 0x3, 0x2, 0x2, 0x2, 0x89, 0x68, 0x3, 0x2, 0x2, 0x2, 
       0x89, 0x70, 0x3, 0x2, 0x2, 0x2, 0x89, 0x80, 0x3, 0x2, 0x2, 0x2, 0x89, 
       0x82, 0x3, 0x2, 0x2, 0x2, 0x89, 0x84, 0x3, 0x2, 0x2, 0x2, 0x8a, 0xb, 
       0x3, 0x2, 0x2, 0x2, 0x8b, 0x8d, 0x7, 0x14, 0x2, 0x2, 0x8c, 0x8e, 
       0x9, 0x3, 0x2, 0x2, 0x8d, 0x8c, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x8e, 0x3, 
       0x2, 0x2, 0x2, 0x8e, 0x91, 0x3, 0x2, 0x2, 0x2, 0x8f, 0x91, 0x5, 0x12, 
       0xa, 0x2, 0x90, 0x8b, 0x3, 0x2, 0x2, 0x2, 0x90, 0x8f, 0x3, 0x2, 0x2, 
       0x2, 0x91, 0xd, 0x3, 0x2, 0x2, 0x2, 0x92, 0x93, 0x9, 0x4, 0x2, 0x2, 
       0x93, 0xf, 0x3, 0x2, 0x2, 0x2, 0x94, 0x99, 0x5, 0x12, 0xa, 0x2, 0x95, 
       0x96, 0x7, 0x1b, 0x2, 0x2, 0x96, 0x98, 0x5, 0x12, 0xa, 0x2, 0x97, 
       0x95, 0x3, 0x2, 0x2, 0x2, 0x98, 0x9b, 0x3, 0x2, 0x2, 0x2, 0x99, 0x97, 
       0x3, 0x2, 0x2, 0x2, 0x99, 0x9a, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x11, 0x3, 
       0x2, 0x2, 0x2, 0x9b, 0x99, 0x3, 0x2, 0x2, 0x2, 0x9c, 0x9d, 0x8, 0xa, 
       0x1, 0x2, 0x9d, 0xb0, 0x5, 0x16, 0xc, 0x2, 0x9e, 0x9f, 0x7, 0x1d, 
       0x2, 0x2, 0x9f, 0xb0, 0x5, 0x12, 0xa, 0xf, 0xa0, 0xa1, 0x7, 0x3, 
       0x2, 0x2, 0xa1, 0xa2, 0x5, 0x12, 0xa, 0x2, 0xa2, 0xa3, 0x7, 0x4, 
       0x2, 0x2, 0xa3, 0xb0, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa5, 0x5, 0x14, 
       0xb, 0x2, 0xa5, 0xa6, 0x7, 0x3, 0x2, 0x2, 0xa6, 0xa7, 0x5, 0x12, 
       0xa, 0x2, 0xa7, 0xa8, 0x7, 0x4, 0x2, 0x2, 0xa8, 0xb0, 0x3, 0x2, 0x2, 
       0x2, 0xa9, 0xb0, 0x5, 0x1a, 0xe, 0x2, 0xaa, 0xb0, 0x5, 0x18, 0xd, 
       0x2, 0xab, 0xb0, 0x7, 0x32, 0x2, 0x2, 0xac, 0xb0, 0x7, 0x25, 0x2, 
       0x2, 0xad, 0xae, 0x7, 0x28, 0x2, 0x2, 0xae, 0xb0, 0x5, 0x12, 0xa, 
       0x3, 0xaf, 0x9c, 0x3, 0x2, 0x2, 0x2, 0xaf, 0x9e, 0x3, 0x2, 0x2, 0x2, 
       0xaf, 0xa0, 0x3, 0x2, 0x2, 0x2, 0xaf, 0xa4, 0x3, 0x2, 0x2, 0x2, 0xaf, 
       0xa9, 0x3, 0x2, 0x2, 0x2, 0xaf, 0xaa, 0x3, 0x2, 0x2, 0x2, 0xaf, 0xab, 
       0x3, 0x2, 0x2, 0x2, 0xaf, 0xac, 0x3, 0x2, 0x2, 0x2, 0xaf, 0xad, 0x3, 
       0x2, 0x2, 0x2, 0xb0, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xb1, 0xb2, 0xc, 0x19, 
       0x2, 0x2, 0xb2, 0xb3, 0x7, 0x1c, 0x2, 0x2, 0xb3, 0xf1, 0x5, 0x12, 
       0xa, 0x1a, 0xb4, 0xb5, 0xc, 0x18, 0x2, 0x2, 0xb5, 0xb6, 0x7, 0x1d, 
       0x2, 0x2, 0xb6, 0xf1, 0x5, 0x12, 0xa, 0x19, 0xb7, 0xb8, 0xc, 0x17, 
       0x2, 0x2, 0xb8, 0xb9, 0x7, 0x16, 0x2, 0x2, 0xb9, 0xf1, 0x5, 0x12, 
       0xa, 0x18, 0xba, 0xbb, 0xc, 0x16, 0x2, 0x2, 0xbb, 0xbc, 0x7, 0x1e, 
       0x2, 0x2, 0xbc, 0xf1, 0x5, 0x12, 0xa, 0x17, 0xbd, 0xbe, 0xc, 0x15, 
       0x2, 0x2, 0xbe, 0xbf, 0x7, 0x1f, 0x2, 0x2, 0xbf, 0xf1, 0x5, 0x12, 
       0xa, 0x16, 0xc0, 0xc1, 0xc, 0x14, 0x2, 0x2, 0xc1, 0xc2, 0x7, 0x20, 
       0x2, 0x2, 0xc2, 0xf1, 0x5, 0x12, 0xa, 0x15, 0xc3, 0xc4, 0xc, 0xe, 
       0x2, 0x2, 0xc4, 0xc5, 0x7, 0x22, 0x2, 0x2, 0xc5, 0xf1, 0x5, 0x12, 
       0xa, 0xf, 0xc6, 0xc7, 0xc, 0x5, 0x2, 0x2, 0xc7, 0xc8, 0x7, 0x26, 
       0x2, 0x2, 0xc8, 0xf1, 0x5, 0x12, 0xa, 0x6, 0xc9, 0xca, 0xc, 0x4, 
       0x2, 0x2, 0xca, 0xcb, 0x7, 0x27, 0x2, 0x2, 0xcb, 0xf1, 0x5, 0x12, 
       0xa, 0x5, 0xcc, 0xcd, 0xc, 0x13, 0x2, 0x2, 0xcd, 0xce, 0x7, 0x18, 
       0x2, 0x2, 0xce, 0xcf, 0x5, 0x12, 0xa, 0x2, 0xcf, 0xd0, 0x7, 0x17, 
       0x2, 0x2, 0xd0, 0xf1, 0x3, 0x2, 0x2, 0x2, 0xd1, 0xd2, 0xc, 0x12, 
       0x2, 0x2, 0xd2, 0xd3, 0x7, 0x1f, 0x2, 0x2, 0xd3, 0xd4, 0x5, 0x12, 
       0xa, 0x2, 0xd4, 0xd5, 0x7, 0x3, 0x2, 0x2, 0xd5, 0xd6, 0x5, 0x10, 
       0x9, 0x2, 0xd6, 0xd7, 0x7, 0x4, 0x2, 0x2, 0xd7, 0xf1, 0x3, 0x2, 0x2, 
       0x2, 0xd8, 0xd9, 0xc, 0x11, 0x2, 0x2, 0xd9, 0xda, 0x7, 0x21, 0x2, 
       0x2, 0xda, 0xdb, 0x5, 0x12, 0xa, 0x2, 0xdb, 0xdc, 0x7, 0x3, 0x2, 
       0x2, 0xdc, 0xdd, 0x7, 0x4, 0x2, 0x2, 0xdd, 0xf1, 0x3, 0x2, 0x2, 0x2, 
       0xde, 0xdf, 0xc, 0x10, 0x2, 0x2, 0xdf, 0xe0, 0x7, 0x21, 0x2, 0x2, 
       0xe0, 0xe1, 0x5, 0x12, 0xa, 0x2, 0xe1, 0xe2, 0x7, 0x3, 0x2, 0x2, 
       0xe2, 0xe3, 0x5, 0x10, 0x9, 0x2, 0xe3, 0xe4, 0x7, 0x4, 0x2, 0x2, 
       0xe4, 0xf1, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe6, 0xc, 0xb, 0x2, 0x2, 0xe6, 
       0xe8, 0x7, 0x3, 0x2, 0x2, 0xe7, 0xe9, 0x5, 0x10, 0x9, 0x2, 0xe8, 
       0xe7, 0x3, 0x2, 0x2, 0x2, 0xe8, 0xe9, 0x3, 0x2, 0x2, 0x2, 0xe9, 0xea, 
       0x3, 0x2, 0x2, 0x2, 0xea, 0xf1, 0x7, 0x4, 0x2, 0x2, 0xeb, 0xec, 0xc, 
       0xa, 0x2, 0x2, 0xec, 0xed, 0x7, 0x23, 0x2, 0x2, 0xed, 0xee, 0x5, 
       0x12, 0xa, 0x2, 0xee, 0xef, 0x7, 0x24, 0x2, 0x2, 0xef, 0xf1, 0x3, 
       0x2, 0x2, 0x2, 0xf0, 0xb1, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xb4, 0x3, 0x2, 
       0x2, 0x2, 0xf0, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xba, 0x3, 0x2, 0x2, 
       0x2, 0xf0, 0xbd, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xc0, 0x3, 0x2, 0x2, 0x2, 
       0xf0, 0xc3, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xc6, 0x3, 0x2, 0x2, 0x2, 0xf0, 
       0xc9, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xd1, 
       0x3, 0x2, 0x2, 0x2, 0xf0, 0xd8, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xde, 0x3, 
       0x2, 0x2, 0x2, 0xf0, 0xe5, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xeb, 0x3, 0x2, 
       0x2, 0x2, 0xf1, 0xf4, 0x3, 0x2, 0x2, 0x2, 0xf2, 0xf0, 0x3, 0x2, 0x2, 
       0x2, 0xf2, 0xf3, 0x3, 0x2, 0x2, 0x2, 0xf3, 0x13, 0x3, 0x2, 0x2, 0x2, 
       0xf4, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xf6, 0x9, 0x5, 0x2, 0x2, 0xf6, 
       0x15, 0x3, 0x2, 0x2, 0x2, 0xf7, 0xf8, 0x7, 0x30, 0x2, 0x2, 0xf8, 
       0x17, 0x3, 0x2, 0x2, 0x2, 0xf9, 0xfa, 0x7, 0x31, 0x2, 0x2, 0xfa, 
       0x19, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfc, 0x7, 0x33, 0x2, 0x2, 0xfc, 
       0x1b, 0x3, 0x2, 0x2, 0x2, 0x15, 0x1e, 0x22, 0x2d, 0x33, 0x4a, 0x52, 
       0x56, 0x5c, 0x66, 0x6d, 0x70, 0x89, 0x8d, 0x90, 0x99, 0xaf, 0xe8, 
       0xf0, 0xf2, 
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

xasm_singleParser::Initializer xasm_singleParser::_init;

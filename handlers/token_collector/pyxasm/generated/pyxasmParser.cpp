
// Generated from pyxasm.g4 by ANTLR 4.8


#include "pyxasmListener.h"
#include "pyxasmVisitor.h"

#include "pyxasmParser.h"


using namespace antlrcpp;
using namespace pyxasm;
using namespace antlr4;

pyxasmParser::pyxasmParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

pyxasmParser::~pyxasmParser() {
  delete _interpreter;
}

std::string pyxasmParser::getGrammarFileName() const {
  return "pyxasm.g4";
}

const std::vector<std::string>& pyxasmParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& pyxasmParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- Single_inputContext ------------------------------------------------------------------

pyxasmParser::Single_inputContext::Single_inputContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Single_inputContext::NEWLINE() {
  return getToken(pyxasmParser::NEWLINE, 0);
}

pyxasmParser::Simple_stmtContext* pyxasmParser::Single_inputContext::simple_stmt() {
  return getRuleContext<pyxasmParser::Simple_stmtContext>(0);
}

pyxasmParser::Compound_stmtContext* pyxasmParser::Single_inputContext::compound_stmt() {
  return getRuleContext<pyxasmParser::Compound_stmtContext>(0);
}


size_t pyxasmParser::Single_inputContext::getRuleIndex() const {
  return pyxasmParser::RuleSingle_input;
}

void pyxasmParser::Single_inputContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_input(this);
}

void pyxasmParser::Single_inputContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_input(this);
}


antlrcpp::Any pyxasmParser::Single_inputContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitSingle_input(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Single_inputContext* pyxasmParser::single_input() {
  Single_inputContext *_localctx = _tracker.createInstance<Single_inputContext>(_ctx, getState());
  enterRule(_localctx, 0, pyxasmParser::RuleSingle_input);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(177);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::NEWLINE: {
        enterOuterAlt(_localctx, 1);
        setState(172);
        match(pyxasmParser::NEWLINE);
        break;
      }

      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::RETURN:
      case pyxasmParser::RAISE:
      case pyxasmParser::FROM:
      case pyxasmParser::IMPORT:
      case pyxasmParser::GLOBAL:
      case pyxasmParser::NONLOCAL:
      case pyxasmParser::ASSERT:
      case pyxasmParser::LAMBDA:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::YIELD:
      case pyxasmParser::DEL:
      case pyxasmParser::PASS:
      case pyxasmParser::CONTINUE:
      case pyxasmParser::BREAK:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::STAR:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 2);
        setState(173);
        simple_stmt();
        break;
      }

      case pyxasmParser::DEF:
      case pyxasmParser::IF:
      case pyxasmParser::WHILE:
      case pyxasmParser::FOR:
      case pyxasmParser::TRY:
      case pyxasmParser::WITH:
      case pyxasmParser::CLASS:
      case pyxasmParser::ASYNC:
      case pyxasmParser::AT: {
        enterOuterAlt(_localctx, 3);
        setState(174);
        compound_stmt();
        setState(175);
        match(pyxasmParser::NEWLINE);
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

//----------------- File_inputContext ------------------------------------------------------------------

pyxasmParser::File_inputContext::File_inputContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::File_inputContext::EOF() {
  return getToken(pyxasmParser::EOF, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::File_inputContext::NEWLINE() {
  return getTokens(pyxasmParser::NEWLINE);
}

tree::TerminalNode* pyxasmParser::File_inputContext::NEWLINE(size_t i) {
  return getToken(pyxasmParser::NEWLINE, i);
}

std::vector<pyxasmParser::StmtContext *> pyxasmParser::File_inputContext::stmt() {
  return getRuleContexts<pyxasmParser::StmtContext>();
}

pyxasmParser::StmtContext* pyxasmParser::File_inputContext::stmt(size_t i) {
  return getRuleContext<pyxasmParser::StmtContext>(i);
}


size_t pyxasmParser::File_inputContext::getRuleIndex() const {
  return pyxasmParser::RuleFile_input;
}

void pyxasmParser::File_inputContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFile_input(this);
}

void pyxasmParser::File_inputContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFile_input(this);
}


antlrcpp::Any pyxasmParser::File_inputContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitFile_input(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::File_inputContext* pyxasmParser::file_input() {
  File_inputContext *_localctx = _tracker.createInstance<File_inputContext>(_ctx, getState());
  enterRule(_localctx, 2, pyxasmParser::RuleFile_input);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(183);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
      | (1ULL << pyxasmParser::NUMBER)
      | (1ULL << pyxasmParser::DEF)
      | (1ULL << pyxasmParser::RETURN)
      | (1ULL << pyxasmParser::RAISE)
      | (1ULL << pyxasmParser::FROM)
      | (1ULL << pyxasmParser::IMPORT)
      | (1ULL << pyxasmParser::GLOBAL)
      | (1ULL << pyxasmParser::NONLOCAL)
      | (1ULL << pyxasmParser::ASSERT)
      | (1ULL << pyxasmParser::IF)
      | (1ULL << pyxasmParser::WHILE)
      | (1ULL << pyxasmParser::FOR)
      | (1ULL << pyxasmParser::TRY)
      | (1ULL << pyxasmParser::WITH)
      | (1ULL << pyxasmParser::LAMBDA)
      | (1ULL << pyxasmParser::NOT)
      | (1ULL << pyxasmParser::NONE)
      | (1ULL << pyxasmParser::TRUE)
      | (1ULL << pyxasmParser::FALSE)
      | (1ULL << pyxasmParser::CLASS)
      | (1ULL << pyxasmParser::YIELD)
      | (1ULL << pyxasmParser::DEL)
      | (1ULL << pyxasmParser::PASS)
      | (1ULL << pyxasmParser::CONTINUE)
      | (1ULL << pyxasmParser::BREAK)
      | (1ULL << pyxasmParser::ASYNC)
      | (1ULL << pyxasmParser::AWAIT)
      | (1ULL << pyxasmParser::NEWLINE)
      | (1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::ELLIPSIS)
      | (1ULL << pyxasmParser::STAR)
      | (1ULL << pyxasmParser::OPEN_PAREN)
      | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
      | (1ULL << (pyxasmParser::MINUS - 66))
      | (1ULL << (pyxasmParser::NOT_OP - 66))
      | (1ULL << (pyxasmParser::OPEN_BRACE - 66))
      | (1ULL << (pyxasmParser::AT - 66)))) != 0)) {
      setState(181);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case pyxasmParser::NEWLINE: {
          setState(179);
          match(pyxasmParser::NEWLINE);
          break;
        }

        case pyxasmParser::STRING:
        case pyxasmParser::NUMBER:
        case pyxasmParser::DEF:
        case pyxasmParser::RETURN:
        case pyxasmParser::RAISE:
        case pyxasmParser::FROM:
        case pyxasmParser::IMPORT:
        case pyxasmParser::GLOBAL:
        case pyxasmParser::NONLOCAL:
        case pyxasmParser::ASSERT:
        case pyxasmParser::IF:
        case pyxasmParser::WHILE:
        case pyxasmParser::FOR:
        case pyxasmParser::TRY:
        case pyxasmParser::WITH:
        case pyxasmParser::LAMBDA:
        case pyxasmParser::NOT:
        case pyxasmParser::NONE:
        case pyxasmParser::TRUE:
        case pyxasmParser::FALSE:
        case pyxasmParser::CLASS:
        case pyxasmParser::YIELD:
        case pyxasmParser::DEL:
        case pyxasmParser::PASS:
        case pyxasmParser::CONTINUE:
        case pyxasmParser::BREAK:
        case pyxasmParser::ASYNC:
        case pyxasmParser::AWAIT:
        case pyxasmParser::NAME:
        case pyxasmParser::ELLIPSIS:
        case pyxasmParser::STAR:
        case pyxasmParser::OPEN_PAREN:
        case pyxasmParser::OPEN_BRACK:
        case pyxasmParser::ADD:
        case pyxasmParser::MINUS:
        case pyxasmParser::NOT_OP:
        case pyxasmParser::OPEN_BRACE:
        case pyxasmParser::AT: {
          setState(180);
          stmt();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(185);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(186);
    match(pyxasmParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Eval_inputContext ------------------------------------------------------------------

pyxasmParser::Eval_inputContext::Eval_inputContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::TestlistContext* pyxasmParser::Eval_inputContext::testlist() {
  return getRuleContext<pyxasmParser::TestlistContext>(0);
}

tree::TerminalNode* pyxasmParser::Eval_inputContext::EOF() {
  return getToken(pyxasmParser::EOF, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Eval_inputContext::NEWLINE() {
  return getTokens(pyxasmParser::NEWLINE);
}

tree::TerminalNode* pyxasmParser::Eval_inputContext::NEWLINE(size_t i) {
  return getToken(pyxasmParser::NEWLINE, i);
}


size_t pyxasmParser::Eval_inputContext::getRuleIndex() const {
  return pyxasmParser::RuleEval_input;
}

void pyxasmParser::Eval_inputContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEval_input(this);
}

void pyxasmParser::Eval_inputContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEval_input(this);
}


antlrcpp::Any pyxasmParser::Eval_inputContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitEval_input(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Eval_inputContext* pyxasmParser::eval_input() {
  Eval_inputContext *_localctx = _tracker.createInstance<Eval_inputContext>(_ctx, getState());
  enterRule(_localctx, 4, pyxasmParser::RuleEval_input);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(188);
    testlist();
    setState(192);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::NEWLINE) {
      setState(189);
      match(pyxasmParser::NEWLINE);
      setState(194);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(195);
    match(pyxasmParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DecoratorContext ------------------------------------------------------------------

pyxasmParser::DecoratorContext::DecoratorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::DecoratorContext::AT() {
  return getToken(pyxasmParser::AT, 0);
}

pyxasmParser::Dotted_nameContext* pyxasmParser::DecoratorContext::dotted_name() {
  return getRuleContext<pyxasmParser::Dotted_nameContext>(0);
}

tree::TerminalNode* pyxasmParser::DecoratorContext::NEWLINE() {
  return getToken(pyxasmParser::NEWLINE, 0);
}

tree::TerminalNode* pyxasmParser::DecoratorContext::OPEN_PAREN() {
  return getToken(pyxasmParser::OPEN_PAREN, 0);
}

tree::TerminalNode* pyxasmParser::DecoratorContext::CLOSE_PAREN() {
  return getToken(pyxasmParser::CLOSE_PAREN, 0);
}

pyxasmParser::ArglistContext* pyxasmParser::DecoratorContext::arglist() {
  return getRuleContext<pyxasmParser::ArglistContext>(0);
}


size_t pyxasmParser::DecoratorContext::getRuleIndex() const {
  return pyxasmParser::RuleDecorator;
}

void pyxasmParser::DecoratorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecorator(this);
}

void pyxasmParser::DecoratorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecorator(this);
}


antlrcpp::Any pyxasmParser::DecoratorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDecorator(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::DecoratorContext* pyxasmParser::decorator() {
  DecoratorContext *_localctx = _tracker.createInstance<DecoratorContext>(_ctx, getState());
  enterRule(_localctx, 6, pyxasmParser::RuleDecorator);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(197);
    match(pyxasmParser::AT);
    setState(198);
    dotted_name();
    setState(204);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::OPEN_PAREN) {
      setState(199);
      match(pyxasmParser::OPEN_PAREN);
      setState(201);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
        | (1ULL << pyxasmParser::NUMBER)
        | (1ULL << pyxasmParser::LAMBDA)
        | (1ULL << pyxasmParser::NOT)
        | (1ULL << pyxasmParser::NONE)
        | (1ULL << pyxasmParser::TRUE)
        | (1ULL << pyxasmParser::FALSE)
        | (1ULL << pyxasmParser::AWAIT)
        | (1ULL << pyxasmParser::NAME)
        | (1ULL << pyxasmParser::ELLIPSIS)
        | (1ULL << pyxasmParser::STAR)
        | (1ULL << pyxasmParser::OPEN_PAREN)
        | (1ULL << pyxasmParser::POWER)
        | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
        | (1ULL << (pyxasmParser::MINUS - 66))
        | (1ULL << (pyxasmParser::NOT_OP - 66))
        | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
        setState(200);
        arglist();
      }
      setState(203);
      match(pyxasmParser::CLOSE_PAREN);
    }
    setState(206);
    match(pyxasmParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DecoratorsContext ------------------------------------------------------------------

pyxasmParser::DecoratorsContext::DecoratorsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::DecoratorContext *> pyxasmParser::DecoratorsContext::decorator() {
  return getRuleContexts<pyxasmParser::DecoratorContext>();
}

pyxasmParser::DecoratorContext* pyxasmParser::DecoratorsContext::decorator(size_t i) {
  return getRuleContext<pyxasmParser::DecoratorContext>(i);
}


size_t pyxasmParser::DecoratorsContext::getRuleIndex() const {
  return pyxasmParser::RuleDecorators;
}

void pyxasmParser::DecoratorsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecorators(this);
}

void pyxasmParser::DecoratorsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecorators(this);
}


antlrcpp::Any pyxasmParser::DecoratorsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDecorators(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::DecoratorsContext* pyxasmParser::decorators() {
  DecoratorsContext *_localctx = _tracker.createInstance<DecoratorsContext>(_ctx, getState());
  enterRule(_localctx, 8, pyxasmParser::RuleDecorators);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(209); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(208);
      decorator();
      setState(211); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == pyxasmParser::AT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DecoratedContext ------------------------------------------------------------------

pyxasmParser::DecoratedContext::DecoratedContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::DecoratorsContext* pyxasmParser::DecoratedContext::decorators() {
  return getRuleContext<pyxasmParser::DecoratorsContext>(0);
}

pyxasmParser::ClassdefContext* pyxasmParser::DecoratedContext::classdef() {
  return getRuleContext<pyxasmParser::ClassdefContext>(0);
}

pyxasmParser::FuncdefContext* pyxasmParser::DecoratedContext::funcdef() {
  return getRuleContext<pyxasmParser::FuncdefContext>(0);
}

pyxasmParser::Async_funcdefContext* pyxasmParser::DecoratedContext::async_funcdef() {
  return getRuleContext<pyxasmParser::Async_funcdefContext>(0);
}


size_t pyxasmParser::DecoratedContext::getRuleIndex() const {
  return pyxasmParser::RuleDecorated;
}

void pyxasmParser::DecoratedContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecorated(this);
}

void pyxasmParser::DecoratedContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecorated(this);
}


antlrcpp::Any pyxasmParser::DecoratedContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDecorated(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::DecoratedContext* pyxasmParser::decorated() {
  DecoratedContext *_localctx = _tracker.createInstance<DecoratedContext>(_ctx, getState());
  enterRule(_localctx, 10, pyxasmParser::RuleDecorated);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(213);
    decorators();
    setState(217);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::CLASS: {
        setState(214);
        classdef();
        break;
      }

      case pyxasmParser::DEF: {
        setState(215);
        funcdef();
        break;
      }

      case pyxasmParser::ASYNC: {
        setState(216);
        async_funcdef();
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

//----------------- Async_funcdefContext ------------------------------------------------------------------

pyxasmParser::Async_funcdefContext::Async_funcdefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Async_funcdefContext::ASYNC() {
  return getToken(pyxasmParser::ASYNC, 0);
}

pyxasmParser::FuncdefContext* pyxasmParser::Async_funcdefContext::funcdef() {
  return getRuleContext<pyxasmParser::FuncdefContext>(0);
}


size_t pyxasmParser::Async_funcdefContext::getRuleIndex() const {
  return pyxasmParser::RuleAsync_funcdef;
}

void pyxasmParser::Async_funcdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAsync_funcdef(this);
}

void pyxasmParser::Async_funcdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAsync_funcdef(this);
}


antlrcpp::Any pyxasmParser::Async_funcdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAsync_funcdef(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Async_funcdefContext* pyxasmParser::async_funcdef() {
  Async_funcdefContext *_localctx = _tracker.createInstance<Async_funcdefContext>(_ctx, getState());
  enterRule(_localctx, 12, pyxasmParser::RuleAsync_funcdef);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(219);
    match(pyxasmParser::ASYNC);
    setState(220);
    funcdef();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncdefContext ------------------------------------------------------------------

pyxasmParser::FuncdefContext::FuncdefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::FuncdefContext::DEF() {
  return getToken(pyxasmParser::DEF, 0);
}

tree::TerminalNode* pyxasmParser::FuncdefContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}

pyxasmParser::ParametersContext* pyxasmParser::FuncdefContext::parameters() {
  return getRuleContext<pyxasmParser::ParametersContext>(0);
}

tree::TerminalNode* pyxasmParser::FuncdefContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::SuiteContext* pyxasmParser::FuncdefContext::suite() {
  return getRuleContext<pyxasmParser::SuiteContext>(0);
}

tree::TerminalNode* pyxasmParser::FuncdefContext::ARROW() {
  return getToken(pyxasmParser::ARROW, 0);
}

pyxasmParser::TestContext* pyxasmParser::FuncdefContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}


size_t pyxasmParser::FuncdefContext::getRuleIndex() const {
  return pyxasmParser::RuleFuncdef;
}

void pyxasmParser::FuncdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFuncdef(this);
}

void pyxasmParser::FuncdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFuncdef(this);
}


antlrcpp::Any pyxasmParser::FuncdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitFuncdef(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::FuncdefContext* pyxasmParser::funcdef() {
  FuncdefContext *_localctx = _tracker.createInstance<FuncdefContext>(_ctx, getState());
  enterRule(_localctx, 14, pyxasmParser::RuleFuncdef);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(222);
    match(pyxasmParser::DEF);
    setState(223);
    match(pyxasmParser::NAME);
    setState(224);
    parameters();
    setState(227);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::ARROW) {
      setState(225);
      match(pyxasmParser::ARROW);
      setState(226);
      test();
    }
    setState(229);
    match(pyxasmParser::COLON);
    setState(230);
    suite();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ParametersContext ------------------------------------------------------------------

pyxasmParser::ParametersContext::ParametersContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::ParametersContext::OPEN_PAREN() {
  return getToken(pyxasmParser::OPEN_PAREN, 0);
}

tree::TerminalNode* pyxasmParser::ParametersContext::CLOSE_PAREN() {
  return getToken(pyxasmParser::CLOSE_PAREN, 0);
}

pyxasmParser::TypedargslistContext* pyxasmParser::ParametersContext::typedargslist() {
  return getRuleContext<pyxasmParser::TypedargslistContext>(0);
}


size_t pyxasmParser::ParametersContext::getRuleIndex() const {
  return pyxasmParser::RuleParameters;
}

void pyxasmParser::ParametersContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParameters(this);
}

void pyxasmParser::ParametersContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParameters(this);
}


antlrcpp::Any pyxasmParser::ParametersContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitParameters(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::ParametersContext* pyxasmParser::parameters() {
  ParametersContext *_localctx = _tracker.createInstance<ParametersContext>(_ctx, getState());
  enterRule(_localctx, 16, pyxasmParser::RuleParameters);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(232);
    match(pyxasmParser::OPEN_PAREN);
    setState(234);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::STAR)
      | (1ULL << pyxasmParser::POWER))) != 0)) {
      setState(233);
      typedargslist();
    }
    setState(236);
    match(pyxasmParser::CLOSE_PAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypedargslistContext ------------------------------------------------------------------

pyxasmParser::TypedargslistContext::TypedargslistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TfpdefContext *> pyxasmParser::TypedargslistContext::tfpdef() {
  return getRuleContexts<pyxasmParser::TfpdefContext>();
}

pyxasmParser::TfpdefContext* pyxasmParser::TypedargslistContext::tfpdef(size_t i) {
  return getRuleContext<pyxasmParser::TfpdefContext>(i);
}

tree::TerminalNode* pyxasmParser::TypedargslistContext::STAR() {
  return getToken(pyxasmParser::STAR, 0);
}

tree::TerminalNode* pyxasmParser::TypedargslistContext::POWER() {
  return getToken(pyxasmParser::POWER, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::TypedargslistContext::ASSIGN() {
  return getTokens(pyxasmParser::ASSIGN);
}

tree::TerminalNode* pyxasmParser::TypedargslistContext::ASSIGN(size_t i) {
  return getToken(pyxasmParser::ASSIGN, i);
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::TypedargslistContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::TypedargslistContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::TypedargslistContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::TypedargslistContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::TypedargslistContext::getRuleIndex() const {
  return pyxasmParser::RuleTypedargslist;
}

void pyxasmParser::TypedargslistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypedargslist(this);
}

void pyxasmParser::TypedargslistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypedargslist(this);
}


antlrcpp::Any pyxasmParser::TypedargslistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTypedargslist(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::TypedargslistContext* pyxasmParser::typedargslist() {
  TypedargslistContext *_localctx = _tracker.createInstance<TypedargslistContext>(_ctx, getState());
  enterRule(_localctx, 18, pyxasmParser::RuleTypedargslist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(319);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::NAME: {
        setState(238);
        tfpdef();
        setState(241);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::ASSIGN) {
          setState(239);
          match(pyxasmParser::ASSIGN);
          setState(240);
          test();
        }
        setState(251);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(243);
            match(pyxasmParser::COMMA);
            setState(244);
            tfpdef();
            setState(247);
            _errHandler->sync(this);

            _la = _input->LA(1);
            if (_la == pyxasmParser::ASSIGN) {
              setState(245);
              match(pyxasmParser::ASSIGN);
              setState(246);
              test();
            } 
          }
          setState(253);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
        }
        setState(287);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::COMMA) {
          setState(254);
          match(pyxasmParser::COMMA);
          setState(285);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
            case pyxasmParser::STAR: {
              setState(255);
              match(pyxasmParser::STAR);
              setState(257);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == pyxasmParser::NAME) {
                setState(256);
                tfpdef();
              }
              setState(267);
              _errHandler->sync(this);
              alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx);
              while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
                if (alt == 1) {
                  setState(259);
                  match(pyxasmParser::COMMA);
                  setState(260);
                  tfpdef();
                  setState(263);
                  _errHandler->sync(this);

                  _la = _input->LA(1);
                  if (_la == pyxasmParser::ASSIGN) {
                    setState(261);
                    match(pyxasmParser::ASSIGN);
                    setState(262);
                    test();
                  } 
                }
                setState(269);
                _errHandler->sync(this);
                alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx);
              }
              setState(278);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == pyxasmParser::COMMA) {
                setState(270);
                match(pyxasmParser::COMMA);
                setState(276);
                _errHandler->sync(this);

                _la = _input->LA(1);
                if (_la == pyxasmParser::POWER) {
                  setState(271);
                  match(pyxasmParser::POWER);
                  setState(272);
                  tfpdef();
                  setState(274);
                  _errHandler->sync(this);

                  _la = _input->LA(1);
                  if (_la == pyxasmParser::COMMA) {
                    setState(273);
                    match(pyxasmParser::COMMA);
                  }
                }
              }
              break;
            }

            case pyxasmParser::POWER: {
              setState(280);
              match(pyxasmParser::POWER);
              setState(281);
              tfpdef();
              setState(283);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == pyxasmParser::COMMA) {
                setState(282);
                match(pyxasmParser::COMMA);
              }
              break;
            }

            case pyxasmParser::CLOSE_PAREN: {
              break;
            }

          default:
            break;
          }
        }
        break;
      }

      case pyxasmParser::STAR: {
        setState(289);
        match(pyxasmParser::STAR);
        setState(291);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::NAME) {
          setState(290);
          tfpdef();
        }
        setState(301);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(293);
            match(pyxasmParser::COMMA);
            setState(294);
            tfpdef();
            setState(297);
            _errHandler->sync(this);

            _la = _input->LA(1);
            if (_la == pyxasmParser::ASSIGN) {
              setState(295);
              match(pyxasmParser::ASSIGN);
              setState(296);
              test();
            } 
          }
          setState(303);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
        }
        setState(312);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::COMMA) {
          setState(304);
          match(pyxasmParser::COMMA);
          setState(310);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == pyxasmParser::POWER) {
            setState(305);
            match(pyxasmParser::POWER);
            setState(306);
            tfpdef();
            setState(308);
            _errHandler->sync(this);

            _la = _input->LA(1);
            if (_la == pyxasmParser::COMMA) {
              setState(307);
              match(pyxasmParser::COMMA);
            }
          }
        }
        break;
      }

      case pyxasmParser::POWER: {
        setState(314);
        match(pyxasmParser::POWER);
        setState(315);
        tfpdef();
        setState(317);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::COMMA) {
          setState(316);
          match(pyxasmParser::COMMA);
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

//----------------- TfpdefContext ------------------------------------------------------------------

pyxasmParser::TfpdefContext::TfpdefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::TfpdefContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}

tree::TerminalNode* pyxasmParser::TfpdefContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::TestContext* pyxasmParser::TfpdefContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}


size_t pyxasmParser::TfpdefContext::getRuleIndex() const {
  return pyxasmParser::RuleTfpdef;
}

void pyxasmParser::TfpdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTfpdef(this);
}

void pyxasmParser::TfpdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTfpdef(this);
}


antlrcpp::Any pyxasmParser::TfpdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTfpdef(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::TfpdefContext* pyxasmParser::tfpdef() {
  TfpdefContext *_localctx = _tracker.createInstance<TfpdefContext>(_ctx, getState());
  enterRule(_localctx, 20, pyxasmParser::RuleTfpdef);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(321);
    match(pyxasmParser::NAME);
    setState(324);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COLON) {
      setState(322);
      match(pyxasmParser::COLON);
      setState(323);
      test();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarargslistContext ------------------------------------------------------------------

pyxasmParser::VarargslistContext::VarargslistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::VfpdefContext *> pyxasmParser::VarargslistContext::vfpdef() {
  return getRuleContexts<pyxasmParser::VfpdefContext>();
}

pyxasmParser::VfpdefContext* pyxasmParser::VarargslistContext::vfpdef(size_t i) {
  return getRuleContext<pyxasmParser::VfpdefContext>(i);
}

tree::TerminalNode* pyxasmParser::VarargslistContext::STAR() {
  return getToken(pyxasmParser::STAR, 0);
}

tree::TerminalNode* pyxasmParser::VarargslistContext::POWER() {
  return getToken(pyxasmParser::POWER, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::VarargslistContext::ASSIGN() {
  return getTokens(pyxasmParser::ASSIGN);
}

tree::TerminalNode* pyxasmParser::VarargslistContext::ASSIGN(size_t i) {
  return getToken(pyxasmParser::ASSIGN, i);
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::VarargslistContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::VarargslistContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::VarargslistContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::VarargslistContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::VarargslistContext::getRuleIndex() const {
  return pyxasmParser::RuleVarargslist;
}

void pyxasmParser::VarargslistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVarargslist(this);
}

void pyxasmParser::VarargslistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVarargslist(this);
}


antlrcpp::Any pyxasmParser::VarargslistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitVarargslist(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::VarargslistContext* pyxasmParser::varargslist() {
  VarargslistContext *_localctx = _tracker.createInstance<VarargslistContext>(_ctx, getState());
  enterRule(_localctx, 22, pyxasmParser::RuleVarargslist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(407);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::NAME: {
        setState(326);
        vfpdef();
        setState(329);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::ASSIGN) {
          setState(327);
          match(pyxasmParser::ASSIGN);
          setState(328);
          test();
        }
        setState(339);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(331);
            match(pyxasmParser::COMMA);
            setState(332);
            vfpdef();
            setState(335);
            _errHandler->sync(this);

            _la = _input->LA(1);
            if (_la == pyxasmParser::ASSIGN) {
              setState(333);
              match(pyxasmParser::ASSIGN);
              setState(334);
              test();
            } 
          }
          setState(341);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx);
        }
        setState(375);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::COMMA) {
          setState(342);
          match(pyxasmParser::COMMA);
          setState(373);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
            case pyxasmParser::STAR: {
              setState(343);
              match(pyxasmParser::STAR);
              setState(345);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == pyxasmParser::NAME) {
                setState(344);
                vfpdef();
              }
              setState(355);
              _errHandler->sync(this);
              alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 36, _ctx);
              while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
                if (alt == 1) {
                  setState(347);
                  match(pyxasmParser::COMMA);
                  setState(348);
                  vfpdef();
                  setState(351);
                  _errHandler->sync(this);

                  _la = _input->LA(1);
                  if (_la == pyxasmParser::ASSIGN) {
                    setState(349);
                    match(pyxasmParser::ASSIGN);
                    setState(350);
                    test();
                  } 
                }
                setState(357);
                _errHandler->sync(this);
                alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 36, _ctx);
              }
              setState(366);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == pyxasmParser::COMMA) {
                setState(358);
                match(pyxasmParser::COMMA);
                setState(364);
                _errHandler->sync(this);

                _la = _input->LA(1);
                if (_la == pyxasmParser::POWER) {
                  setState(359);
                  match(pyxasmParser::POWER);
                  setState(360);
                  vfpdef();
                  setState(362);
                  _errHandler->sync(this);

                  _la = _input->LA(1);
                  if (_la == pyxasmParser::COMMA) {
                    setState(361);
                    match(pyxasmParser::COMMA);
                  }
                }
              }
              break;
            }

            case pyxasmParser::POWER: {
              setState(368);
              match(pyxasmParser::POWER);
              setState(369);
              vfpdef();
              setState(371);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == pyxasmParser::COMMA) {
                setState(370);
                match(pyxasmParser::COMMA);
              }
              break;
            }

            case pyxasmParser::COLON: {
              break;
            }

          default:
            break;
          }
        }
        break;
      }

      case pyxasmParser::STAR: {
        setState(377);
        match(pyxasmParser::STAR);
        setState(379);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::NAME) {
          setState(378);
          vfpdef();
        }
        setState(389);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 45, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(381);
            match(pyxasmParser::COMMA);
            setState(382);
            vfpdef();
            setState(385);
            _errHandler->sync(this);

            _la = _input->LA(1);
            if (_la == pyxasmParser::ASSIGN) {
              setState(383);
              match(pyxasmParser::ASSIGN);
              setState(384);
              test();
            } 
          }
          setState(391);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 45, _ctx);
        }
        setState(400);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::COMMA) {
          setState(392);
          match(pyxasmParser::COMMA);
          setState(398);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == pyxasmParser::POWER) {
            setState(393);
            match(pyxasmParser::POWER);
            setState(394);
            vfpdef();
            setState(396);
            _errHandler->sync(this);

            _la = _input->LA(1);
            if (_la == pyxasmParser::COMMA) {
              setState(395);
              match(pyxasmParser::COMMA);
            }
          }
        }
        break;
      }

      case pyxasmParser::POWER: {
        setState(402);
        match(pyxasmParser::POWER);
        setState(403);
        vfpdef();
        setState(405);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::COMMA) {
          setState(404);
          match(pyxasmParser::COMMA);
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

//----------------- VfpdefContext ------------------------------------------------------------------

pyxasmParser::VfpdefContext::VfpdefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::VfpdefContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}


size_t pyxasmParser::VfpdefContext::getRuleIndex() const {
  return pyxasmParser::RuleVfpdef;
}

void pyxasmParser::VfpdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVfpdef(this);
}

void pyxasmParser::VfpdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVfpdef(this);
}


antlrcpp::Any pyxasmParser::VfpdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitVfpdef(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::VfpdefContext* pyxasmParser::vfpdef() {
  VfpdefContext *_localctx = _tracker.createInstance<VfpdefContext>(_ctx, getState());
  enterRule(_localctx, 24, pyxasmParser::RuleVfpdef);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(409);
    match(pyxasmParser::NAME);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StmtContext ------------------------------------------------------------------

pyxasmParser::StmtContext::StmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Simple_stmtContext* pyxasmParser::StmtContext::simple_stmt() {
  return getRuleContext<pyxasmParser::Simple_stmtContext>(0);
}

pyxasmParser::Compound_stmtContext* pyxasmParser::StmtContext::compound_stmt() {
  return getRuleContext<pyxasmParser::Compound_stmtContext>(0);
}


size_t pyxasmParser::StmtContext::getRuleIndex() const {
  return pyxasmParser::RuleStmt;
}

void pyxasmParser::StmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStmt(this);
}

void pyxasmParser::StmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStmt(this);
}


antlrcpp::Any pyxasmParser::StmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitStmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::StmtContext* pyxasmParser::stmt() {
  StmtContext *_localctx = _tracker.createInstance<StmtContext>(_ctx, getState());
  enterRule(_localctx, 26, pyxasmParser::RuleStmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(413);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::RETURN:
      case pyxasmParser::RAISE:
      case pyxasmParser::FROM:
      case pyxasmParser::IMPORT:
      case pyxasmParser::GLOBAL:
      case pyxasmParser::NONLOCAL:
      case pyxasmParser::ASSERT:
      case pyxasmParser::LAMBDA:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::YIELD:
      case pyxasmParser::DEL:
      case pyxasmParser::PASS:
      case pyxasmParser::CONTINUE:
      case pyxasmParser::BREAK:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::STAR:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 1);
        setState(411);
        simple_stmt();
        break;
      }

      case pyxasmParser::DEF:
      case pyxasmParser::IF:
      case pyxasmParser::WHILE:
      case pyxasmParser::FOR:
      case pyxasmParser::TRY:
      case pyxasmParser::WITH:
      case pyxasmParser::CLASS:
      case pyxasmParser::ASYNC:
      case pyxasmParser::AT: {
        enterOuterAlt(_localctx, 2);
        setState(412);
        compound_stmt();
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

//----------------- Simple_stmtContext ------------------------------------------------------------------

pyxasmParser::Simple_stmtContext::Simple_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Small_stmtContext *> pyxasmParser::Simple_stmtContext::small_stmt() {
  return getRuleContexts<pyxasmParser::Small_stmtContext>();
}

pyxasmParser::Small_stmtContext* pyxasmParser::Simple_stmtContext::small_stmt(size_t i) {
  return getRuleContext<pyxasmParser::Small_stmtContext>(i);
}

tree::TerminalNode* pyxasmParser::Simple_stmtContext::NEWLINE() {
  return getToken(pyxasmParser::NEWLINE, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Simple_stmtContext::SEMI_COLON() {
  return getTokens(pyxasmParser::SEMI_COLON);
}

tree::TerminalNode* pyxasmParser::Simple_stmtContext::SEMI_COLON(size_t i) {
  return getToken(pyxasmParser::SEMI_COLON, i);
}


size_t pyxasmParser::Simple_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleSimple_stmt;
}

void pyxasmParser::Simple_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSimple_stmt(this);
}

void pyxasmParser::Simple_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSimple_stmt(this);
}


antlrcpp::Any pyxasmParser::Simple_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitSimple_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Simple_stmtContext* pyxasmParser::simple_stmt() {
  Simple_stmtContext *_localctx = _tracker.createInstance<Simple_stmtContext>(_ctx, getState());
  enterRule(_localctx, 28, pyxasmParser::RuleSimple_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(415);
    small_stmt();
    setState(420);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(416);
        match(pyxasmParser::SEMI_COLON);
        setState(417);
        small_stmt(); 
      }
      setState(422);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx);
    }
    setState(424);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::SEMI_COLON) {
      setState(423);
      match(pyxasmParser::SEMI_COLON);
    }
    setState(426);
    match(pyxasmParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Small_stmtContext ------------------------------------------------------------------

pyxasmParser::Small_stmtContext::Small_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Expr_stmtContext* pyxasmParser::Small_stmtContext::expr_stmt() {
  return getRuleContext<pyxasmParser::Expr_stmtContext>(0);
}

pyxasmParser::Del_stmtContext* pyxasmParser::Small_stmtContext::del_stmt() {
  return getRuleContext<pyxasmParser::Del_stmtContext>(0);
}

pyxasmParser::Pass_stmtContext* pyxasmParser::Small_stmtContext::pass_stmt() {
  return getRuleContext<pyxasmParser::Pass_stmtContext>(0);
}

pyxasmParser::Flow_stmtContext* pyxasmParser::Small_stmtContext::flow_stmt() {
  return getRuleContext<pyxasmParser::Flow_stmtContext>(0);
}

pyxasmParser::Import_stmtContext* pyxasmParser::Small_stmtContext::import_stmt() {
  return getRuleContext<pyxasmParser::Import_stmtContext>(0);
}

pyxasmParser::Global_stmtContext* pyxasmParser::Small_stmtContext::global_stmt() {
  return getRuleContext<pyxasmParser::Global_stmtContext>(0);
}

pyxasmParser::Nonlocal_stmtContext* pyxasmParser::Small_stmtContext::nonlocal_stmt() {
  return getRuleContext<pyxasmParser::Nonlocal_stmtContext>(0);
}

pyxasmParser::Assert_stmtContext* pyxasmParser::Small_stmtContext::assert_stmt() {
  return getRuleContext<pyxasmParser::Assert_stmtContext>(0);
}


size_t pyxasmParser::Small_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleSmall_stmt;
}

void pyxasmParser::Small_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSmall_stmt(this);
}

void pyxasmParser::Small_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSmall_stmt(this);
}


antlrcpp::Any pyxasmParser::Small_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitSmall_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Small_stmtContext* pyxasmParser::small_stmt() {
  Small_stmtContext *_localctx = _tracker.createInstance<Small_stmtContext>(_ctx, getState());
  enterRule(_localctx, 30, pyxasmParser::RuleSmall_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(436);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::LAMBDA:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::STAR:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        setState(428);
        expr_stmt();
        break;
      }

      case pyxasmParser::DEL: {
        setState(429);
        del_stmt();
        break;
      }

      case pyxasmParser::PASS: {
        setState(430);
        pass_stmt();
        break;
      }

      case pyxasmParser::RETURN:
      case pyxasmParser::RAISE:
      case pyxasmParser::YIELD:
      case pyxasmParser::CONTINUE:
      case pyxasmParser::BREAK: {
        setState(431);
        flow_stmt();
        break;
      }

      case pyxasmParser::FROM:
      case pyxasmParser::IMPORT: {
        setState(432);
        import_stmt();
        break;
      }

      case pyxasmParser::GLOBAL: {
        setState(433);
        global_stmt();
        break;
      }

      case pyxasmParser::NONLOCAL: {
        setState(434);
        nonlocal_stmt();
        break;
      }

      case pyxasmParser::ASSERT: {
        setState(435);
        assert_stmt();
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

//----------------- Expr_stmtContext ------------------------------------------------------------------

pyxasmParser::Expr_stmtContext::Expr_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Testlist_star_exprContext *> pyxasmParser::Expr_stmtContext::testlist_star_expr() {
  return getRuleContexts<pyxasmParser::Testlist_star_exprContext>();
}

pyxasmParser::Testlist_star_exprContext* pyxasmParser::Expr_stmtContext::testlist_star_expr(size_t i) {
  return getRuleContext<pyxasmParser::Testlist_star_exprContext>(i);
}

pyxasmParser::AnnassignContext* pyxasmParser::Expr_stmtContext::annassign() {
  return getRuleContext<pyxasmParser::AnnassignContext>(0);
}

pyxasmParser::AugassignContext* pyxasmParser::Expr_stmtContext::augassign() {
  return getRuleContext<pyxasmParser::AugassignContext>(0);
}

std::vector<pyxasmParser::Yield_exprContext *> pyxasmParser::Expr_stmtContext::yield_expr() {
  return getRuleContexts<pyxasmParser::Yield_exprContext>();
}

pyxasmParser::Yield_exprContext* pyxasmParser::Expr_stmtContext::yield_expr(size_t i) {
  return getRuleContext<pyxasmParser::Yield_exprContext>(i);
}

pyxasmParser::TestlistContext* pyxasmParser::Expr_stmtContext::testlist() {
  return getRuleContext<pyxasmParser::TestlistContext>(0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Expr_stmtContext::ASSIGN() {
  return getTokens(pyxasmParser::ASSIGN);
}

tree::TerminalNode* pyxasmParser::Expr_stmtContext::ASSIGN(size_t i) {
  return getToken(pyxasmParser::ASSIGN, i);
}


size_t pyxasmParser::Expr_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleExpr_stmt;
}

void pyxasmParser::Expr_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpr_stmt(this);
}

void pyxasmParser::Expr_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpr_stmt(this);
}


antlrcpp::Any pyxasmParser::Expr_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitExpr_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Expr_stmtContext* pyxasmParser::expr_stmt() {
  Expr_stmtContext *_localctx = _tracker.createInstance<Expr_stmtContext>(_ctx, getState());
  enterRule(_localctx, 32, pyxasmParser::RuleExpr_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(438);
    testlist_star_expr();
    setState(455);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::COLON: {
        setState(439);
        annassign();
        break;
      }

      case pyxasmParser::ADD_ASSIGN:
      case pyxasmParser::SUB_ASSIGN:
      case pyxasmParser::MULT_ASSIGN:
      case pyxasmParser::AT_ASSIGN:
      case pyxasmParser::DIV_ASSIGN:
      case pyxasmParser::MOD_ASSIGN:
      case pyxasmParser::AND_ASSIGN:
      case pyxasmParser::OR_ASSIGN:
      case pyxasmParser::XOR_ASSIGN:
      case pyxasmParser::LEFT_SHIFT_ASSIGN:
      case pyxasmParser::RIGHT_SHIFT_ASSIGN:
      case pyxasmParser::POWER_ASSIGN:
      case pyxasmParser::IDIV_ASSIGN: {
        setState(440);
        augassign();
        setState(443);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case pyxasmParser::YIELD: {
            setState(441);
            yield_expr();
            break;
          }

          case pyxasmParser::STRING:
          case pyxasmParser::NUMBER:
          case pyxasmParser::LAMBDA:
          case pyxasmParser::NOT:
          case pyxasmParser::NONE:
          case pyxasmParser::TRUE:
          case pyxasmParser::FALSE:
          case pyxasmParser::AWAIT:
          case pyxasmParser::NAME:
          case pyxasmParser::ELLIPSIS:
          case pyxasmParser::OPEN_PAREN:
          case pyxasmParser::OPEN_BRACK:
          case pyxasmParser::ADD:
          case pyxasmParser::MINUS:
          case pyxasmParser::NOT_OP:
          case pyxasmParser::OPEN_BRACE: {
            setState(442);
            testlist();
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        break;
      }

      case pyxasmParser::NEWLINE:
      case pyxasmParser::SEMI_COLON:
      case pyxasmParser::ASSIGN: {
        setState(452);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == pyxasmParser::ASSIGN) {
          setState(445);
          match(pyxasmParser::ASSIGN);
          setState(448);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
            case pyxasmParser::YIELD: {
              setState(446);
              yield_expr();
              break;
            }

            case pyxasmParser::STRING:
            case pyxasmParser::NUMBER:
            case pyxasmParser::LAMBDA:
            case pyxasmParser::NOT:
            case pyxasmParser::NONE:
            case pyxasmParser::TRUE:
            case pyxasmParser::FALSE:
            case pyxasmParser::AWAIT:
            case pyxasmParser::NAME:
            case pyxasmParser::ELLIPSIS:
            case pyxasmParser::STAR:
            case pyxasmParser::OPEN_PAREN:
            case pyxasmParser::OPEN_BRACK:
            case pyxasmParser::ADD:
            case pyxasmParser::MINUS:
            case pyxasmParser::NOT_OP:
            case pyxasmParser::OPEN_BRACE: {
              setState(447);
              testlist_star_expr();
              break;
            }

          default:
            throw NoViableAltException(this);
          }
          setState(454);
          _errHandler->sync(this);
          _la = _input->LA(1);
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

//----------------- AnnassignContext ------------------------------------------------------------------

pyxasmParser::AnnassignContext::AnnassignContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::AnnassignContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::AnnassignContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::AnnassignContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

tree::TerminalNode* pyxasmParser::AnnassignContext::ASSIGN() {
  return getToken(pyxasmParser::ASSIGN, 0);
}


size_t pyxasmParser::AnnassignContext::getRuleIndex() const {
  return pyxasmParser::RuleAnnassign;
}

void pyxasmParser::AnnassignContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnnassign(this);
}

void pyxasmParser::AnnassignContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnnassign(this);
}


antlrcpp::Any pyxasmParser::AnnassignContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAnnassign(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::AnnassignContext* pyxasmParser::annassign() {
  AnnassignContext *_localctx = _tracker.createInstance<AnnassignContext>(_ctx, getState());
  enterRule(_localctx, 34, pyxasmParser::RuleAnnassign);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(457);
    match(pyxasmParser::COLON);
    setState(458);
    test();
    setState(461);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::ASSIGN) {
      setState(459);
      match(pyxasmParser::ASSIGN);
      setState(460);
      test();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Testlist_star_exprContext ------------------------------------------------------------------

pyxasmParser::Testlist_star_exprContext::Testlist_star_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::Testlist_star_exprContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::Testlist_star_exprContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

std::vector<pyxasmParser::Star_exprContext *> pyxasmParser::Testlist_star_exprContext::star_expr() {
  return getRuleContexts<pyxasmParser::Star_exprContext>();
}

pyxasmParser::Star_exprContext* pyxasmParser::Testlist_star_exprContext::star_expr(size_t i) {
  return getRuleContext<pyxasmParser::Star_exprContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Testlist_star_exprContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::Testlist_star_exprContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::Testlist_star_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleTestlist_star_expr;
}

void pyxasmParser::Testlist_star_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTestlist_star_expr(this);
}

void pyxasmParser::Testlist_star_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTestlist_star_expr(this);
}


antlrcpp::Any pyxasmParser::Testlist_star_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTestlist_star_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Testlist_star_exprContext* pyxasmParser::testlist_star_expr() {
  Testlist_star_exprContext *_localctx = _tracker.createInstance<Testlist_star_exprContext>(_ctx, getState());
  enterRule(_localctx, 36, pyxasmParser::RuleTestlist_star_expr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(465);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::LAMBDA:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        setState(463);
        test();
        break;
      }

      case pyxasmParser::STAR: {
        setState(464);
        star_expr();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(474);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 62, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(467);
        match(pyxasmParser::COMMA);
        setState(470);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case pyxasmParser::STRING:
          case pyxasmParser::NUMBER:
          case pyxasmParser::LAMBDA:
          case pyxasmParser::NOT:
          case pyxasmParser::NONE:
          case pyxasmParser::TRUE:
          case pyxasmParser::FALSE:
          case pyxasmParser::AWAIT:
          case pyxasmParser::NAME:
          case pyxasmParser::ELLIPSIS:
          case pyxasmParser::OPEN_PAREN:
          case pyxasmParser::OPEN_BRACK:
          case pyxasmParser::ADD:
          case pyxasmParser::MINUS:
          case pyxasmParser::NOT_OP:
          case pyxasmParser::OPEN_BRACE: {
            setState(468);
            test();
            break;
          }

          case pyxasmParser::STAR: {
            setState(469);
            star_expr();
            break;
          }

        default:
          throw NoViableAltException(this);
        } 
      }
      setState(476);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 62, _ctx);
    }
    setState(478);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COMMA) {
      setState(477);
      match(pyxasmParser::COMMA);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AugassignContext ------------------------------------------------------------------

pyxasmParser::AugassignContext::AugassignContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::AugassignContext::ADD_ASSIGN() {
  return getToken(pyxasmParser::ADD_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::SUB_ASSIGN() {
  return getToken(pyxasmParser::SUB_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::MULT_ASSIGN() {
  return getToken(pyxasmParser::MULT_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::AT_ASSIGN() {
  return getToken(pyxasmParser::AT_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::DIV_ASSIGN() {
  return getToken(pyxasmParser::DIV_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::MOD_ASSIGN() {
  return getToken(pyxasmParser::MOD_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::AND_ASSIGN() {
  return getToken(pyxasmParser::AND_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::OR_ASSIGN() {
  return getToken(pyxasmParser::OR_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::XOR_ASSIGN() {
  return getToken(pyxasmParser::XOR_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::LEFT_SHIFT_ASSIGN() {
  return getToken(pyxasmParser::LEFT_SHIFT_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::RIGHT_SHIFT_ASSIGN() {
  return getToken(pyxasmParser::RIGHT_SHIFT_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::POWER_ASSIGN() {
  return getToken(pyxasmParser::POWER_ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::AugassignContext::IDIV_ASSIGN() {
  return getToken(pyxasmParser::IDIV_ASSIGN, 0);
}


size_t pyxasmParser::AugassignContext::getRuleIndex() const {
  return pyxasmParser::RuleAugassign;
}

void pyxasmParser::AugassignContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAugassign(this);
}

void pyxasmParser::AugassignContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAugassign(this);
}


antlrcpp::Any pyxasmParser::AugassignContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAugassign(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::AugassignContext* pyxasmParser::augassign() {
  AugassignContext *_localctx = _tracker.createInstance<AugassignContext>(_ctx, getState());
  enterRule(_localctx, 38, pyxasmParser::RuleAugassign);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(480);
    _la = _input->LA(1);
    if (!(((((_la - 83) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 83)) & ((1ULL << (pyxasmParser::ADD_ASSIGN - 83))
      | (1ULL << (pyxasmParser::SUB_ASSIGN - 83))
      | (1ULL << (pyxasmParser::MULT_ASSIGN - 83))
      | (1ULL << (pyxasmParser::AT_ASSIGN - 83))
      | (1ULL << (pyxasmParser::DIV_ASSIGN - 83))
      | (1ULL << (pyxasmParser::MOD_ASSIGN - 83))
      | (1ULL << (pyxasmParser::AND_ASSIGN - 83))
      | (1ULL << (pyxasmParser::OR_ASSIGN - 83))
      | (1ULL << (pyxasmParser::XOR_ASSIGN - 83))
      | (1ULL << (pyxasmParser::LEFT_SHIFT_ASSIGN - 83))
      | (1ULL << (pyxasmParser::RIGHT_SHIFT_ASSIGN - 83))
      | (1ULL << (pyxasmParser::POWER_ASSIGN - 83))
      | (1ULL << (pyxasmParser::IDIV_ASSIGN - 83)))) != 0))) {
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

//----------------- Del_stmtContext ------------------------------------------------------------------

pyxasmParser::Del_stmtContext::Del_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Del_stmtContext::DEL() {
  return getToken(pyxasmParser::DEL, 0);
}

pyxasmParser::ExprlistContext* pyxasmParser::Del_stmtContext::exprlist() {
  return getRuleContext<pyxasmParser::ExprlistContext>(0);
}


size_t pyxasmParser::Del_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleDel_stmt;
}

void pyxasmParser::Del_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDel_stmt(this);
}

void pyxasmParser::Del_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDel_stmt(this);
}


antlrcpp::Any pyxasmParser::Del_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDel_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Del_stmtContext* pyxasmParser::del_stmt() {
  Del_stmtContext *_localctx = _tracker.createInstance<Del_stmtContext>(_ctx, getState());
  enterRule(_localctx, 40, pyxasmParser::RuleDel_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(482);
    match(pyxasmParser::DEL);
    setState(483);
    exprlist();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Pass_stmtContext ------------------------------------------------------------------

pyxasmParser::Pass_stmtContext::Pass_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Pass_stmtContext::PASS() {
  return getToken(pyxasmParser::PASS, 0);
}


size_t pyxasmParser::Pass_stmtContext::getRuleIndex() const {
  return pyxasmParser::RulePass_stmt;
}

void pyxasmParser::Pass_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPass_stmt(this);
}

void pyxasmParser::Pass_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPass_stmt(this);
}


antlrcpp::Any pyxasmParser::Pass_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitPass_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Pass_stmtContext* pyxasmParser::pass_stmt() {
  Pass_stmtContext *_localctx = _tracker.createInstance<Pass_stmtContext>(_ctx, getState());
  enterRule(_localctx, 42, pyxasmParser::RulePass_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(485);
    match(pyxasmParser::PASS);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Flow_stmtContext ------------------------------------------------------------------

pyxasmParser::Flow_stmtContext::Flow_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Break_stmtContext* pyxasmParser::Flow_stmtContext::break_stmt() {
  return getRuleContext<pyxasmParser::Break_stmtContext>(0);
}

pyxasmParser::Continue_stmtContext* pyxasmParser::Flow_stmtContext::continue_stmt() {
  return getRuleContext<pyxasmParser::Continue_stmtContext>(0);
}

pyxasmParser::Return_stmtContext* pyxasmParser::Flow_stmtContext::return_stmt() {
  return getRuleContext<pyxasmParser::Return_stmtContext>(0);
}

pyxasmParser::Raise_stmtContext* pyxasmParser::Flow_stmtContext::raise_stmt() {
  return getRuleContext<pyxasmParser::Raise_stmtContext>(0);
}

pyxasmParser::Yield_stmtContext* pyxasmParser::Flow_stmtContext::yield_stmt() {
  return getRuleContext<pyxasmParser::Yield_stmtContext>(0);
}


size_t pyxasmParser::Flow_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleFlow_stmt;
}

void pyxasmParser::Flow_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFlow_stmt(this);
}

void pyxasmParser::Flow_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFlow_stmt(this);
}


antlrcpp::Any pyxasmParser::Flow_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitFlow_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Flow_stmtContext* pyxasmParser::flow_stmt() {
  Flow_stmtContext *_localctx = _tracker.createInstance<Flow_stmtContext>(_ctx, getState());
  enterRule(_localctx, 44, pyxasmParser::RuleFlow_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(492);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::BREAK: {
        enterOuterAlt(_localctx, 1);
        setState(487);
        break_stmt();
        break;
      }

      case pyxasmParser::CONTINUE: {
        enterOuterAlt(_localctx, 2);
        setState(488);
        continue_stmt();
        break;
      }

      case pyxasmParser::RETURN: {
        enterOuterAlt(_localctx, 3);
        setState(489);
        return_stmt();
        break;
      }

      case pyxasmParser::RAISE: {
        enterOuterAlt(_localctx, 4);
        setState(490);
        raise_stmt();
        break;
      }

      case pyxasmParser::YIELD: {
        enterOuterAlt(_localctx, 5);
        setState(491);
        yield_stmt();
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

//----------------- Break_stmtContext ------------------------------------------------------------------

pyxasmParser::Break_stmtContext::Break_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Break_stmtContext::BREAK() {
  return getToken(pyxasmParser::BREAK, 0);
}


size_t pyxasmParser::Break_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleBreak_stmt;
}

void pyxasmParser::Break_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBreak_stmt(this);
}

void pyxasmParser::Break_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBreak_stmt(this);
}


antlrcpp::Any pyxasmParser::Break_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitBreak_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Break_stmtContext* pyxasmParser::break_stmt() {
  Break_stmtContext *_localctx = _tracker.createInstance<Break_stmtContext>(_ctx, getState());
  enterRule(_localctx, 46, pyxasmParser::RuleBreak_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(494);
    match(pyxasmParser::BREAK);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Continue_stmtContext ------------------------------------------------------------------

pyxasmParser::Continue_stmtContext::Continue_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Continue_stmtContext::CONTINUE() {
  return getToken(pyxasmParser::CONTINUE, 0);
}


size_t pyxasmParser::Continue_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleContinue_stmt;
}

void pyxasmParser::Continue_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterContinue_stmt(this);
}

void pyxasmParser::Continue_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitContinue_stmt(this);
}


antlrcpp::Any pyxasmParser::Continue_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitContinue_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Continue_stmtContext* pyxasmParser::continue_stmt() {
  Continue_stmtContext *_localctx = _tracker.createInstance<Continue_stmtContext>(_ctx, getState());
  enterRule(_localctx, 48, pyxasmParser::RuleContinue_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(496);
    match(pyxasmParser::CONTINUE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Return_stmtContext ------------------------------------------------------------------

pyxasmParser::Return_stmtContext::Return_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Return_stmtContext::RETURN() {
  return getToken(pyxasmParser::RETURN, 0);
}

pyxasmParser::TestlistContext* pyxasmParser::Return_stmtContext::testlist() {
  return getRuleContext<pyxasmParser::TestlistContext>(0);
}


size_t pyxasmParser::Return_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleReturn_stmt;
}

void pyxasmParser::Return_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReturn_stmt(this);
}

void pyxasmParser::Return_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReturn_stmt(this);
}


antlrcpp::Any pyxasmParser::Return_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitReturn_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Return_stmtContext* pyxasmParser::return_stmt() {
  Return_stmtContext *_localctx = _tracker.createInstance<Return_stmtContext>(_ctx, getState());
  enterRule(_localctx, 50, pyxasmParser::RuleReturn_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(498);
    match(pyxasmParser::RETURN);
    setState(500);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
      | (1ULL << pyxasmParser::NUMBER)
      | (1ULL << pyxasmParser::LAMBDA)
      | (1ULL << pyxasmParser::NOT)
      | (1ULL << pyxasmParser::NONE)
      | (1ULL << pyxasmParser::TRUE)
      | (1ULL << pyxasmParser::FALSE)
      | (1ULL << pyxasmParser::AWAIT)
      | (1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::ELLIPSIS)
      | (1ULL << pyxasmParser::OPEN_PAREN)
      | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
      | (1ULL << (pyxasmParser::MINUS - 66))
      | (1ULL << (pyxasmParser::NOT_OP - 66))
      | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
      setState(499);
      testlist();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Yield_stmtContext ------------------------------------------------------------------

pyxasmParser::Yield_stmtContext::Yield_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Yield_exprContext* pyxasmParser::Yield_stmtContext::yield_expr() {
  return getRuleContext<pyxasmParser::Yield_exprContext>(0);
}


size_t pyxasmParser::Yield_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleYield_stmt;
}

void pyxasmParser::Yield_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterYield_stmt(this);
}

void pyxasmParser::Yield_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitYield_stmt(this);
}


antlrcpp::Any pyxasmParser::Yield_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitYield_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Yield_stmtContext* pyxasmParser::yield_stmt() {
  Yield_stmtContext *_localctx = _tracker.createInstance<Yield_stmtContext>(_ctx, getState());
  enterRule(_localctx, 52, pyxasmParser::RuleYield_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(502);
    yield_expr();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Raise_stmtContext ------------------------------------------------------------------

pyxasmParser::Raise_stmtContext::Raise_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Raise_stmtContext::RAISE() {
  return getToken(pyxasmParser::RAISE, 0);
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::Raise_stmtContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::Raise_stmtContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

tree::TerminalNode* pyxasmParser::Raise_stmtContext::FROM() {
  return getToken(pyxasmParser::FROM, 0);
}


size_t pyxasmParser::Raise_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleRaise_stmt;
}

void pyxasmParser::Raise_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRaise_stmt(this);
}

void pyxasmParser::Raise_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRaise_stmt(this);
}


antlrcpp::Any pyxasmParser::Raise_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitRaise_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Raise_stmtContext* pyxasmParser::raise_stmt() {
  Raise_stmtContext *_localctx = _tracker.createInstance<Raise_stmtContext>(_ctx, getState());
  enterRule(_localctx, 54, pyxasmParser::RuleRaise_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(504);
    match(pyxasmParser::RAISE);
    setState(510);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
      | (1ULL << pyxasmParser::NUMBER)
      | (1ULL << pyxasmParser::LAMBDA)
      | (1ULL << pyxasmParser::NOT)
      | (1ULL << pyxasmParser::NONE)
      | (1ULL << pyxasmParser::TRUE)
      | (1ULL << pyxasmParser::FALSE)
      | (1ULL << pyxasmParser::AWAIT)
      | (1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::ELLIPSIS)
      | (1ULL << pyxasmParser::OPEN_PAREN)
      | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
      | (1ULL << (pyxasmParser::MINUS - 66))
      | (1ULL << (pyxasmParser::NOT_OP - 66))
      | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
      setState(505);
      test();
      setState(508);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == pyxasmParser::FROM) {
        setState(506);
        match(pyxasmParser::FROM);
        setState(507);
        test();
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

//----------------- Import_stmtContext ------------------------------------------------------------------

pyxasmParser::Import_stmtContext::Import_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Import_nameContext* pyxasmParser::Import_stmtContext::import_name() {
  return getRuleContext<pyxasmParser::Import_nameContext>(0);
}

pyxasmParser::Import_fromContext* pyxasmParser::Import_stmtContext::import_from() {
  return getRuleContext<pyxasmParser::Import_fromContext>(0);
}


size_t pyxasmParser::Import_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleImport_stmt;
}

void pyxasmParser::Import_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImport_stmt(this);
}

void pyxasmParser::Import_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImport_stmt(this);
}


antlrcpp::Any pyxasmParser::Import_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitImport_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Import_stmtContext* pyxasmParser::import_stmt() {
  Import_stmtContext *_localctx = _tracker.createInstance<Import_stmtContext>(_ctx, getState());
  enterRule(_localctx, 56, pyxasmParser::RuleImport_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(514);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::IMPORT: {
        enterOuterAlt(_localctx, 1);
        setState(512);
        import_name();
        break;
      }

      case pyxasmParser::FROM: {
        enterOuterAlt(_localctx, 2);
        setState(513);
        import_from();
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

//----------------- Import_nameContext ------------------------------------------------------------------

pyxasmParser::Import_nameContext::Import_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Import_nameContext::IMPORT() {
  return getToken(pyxasmParser::IMPORT, 0);
}

pyxasmParser::Dotted_as_namesContext* pyxasmParser::Import_nameContext::dotted_as_names() {
  return getRuleContext<pyxasmParser::Dotted_as_namesContext>(0);
}


size_t pyxasmParser::Import_nameContext::getRuleIndex() const {
  return pyxasmParser::RuleImport_name;
}

void pyxasmParser::Import_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImport_name(this);
}

void pyxasmParser::Import_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImport_name(this);
}


antlrcpp::Any pyxasmParser::Import_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitImport_name(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Import_nameContext* pyxasmParser::import_name() {
  Import_nameContext *_localctx = _tracker.createInstance<Import_nameContext>(_ctx, getState());
  enterRule(_localctx, 58, pyxasmParser::RuleImport_name);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(516);
    match(pyxasmParser::IMPORT);
    setState(517);
    dotted_as_names();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Import_fromContext ------------------------------------------------------------------

pyxasmParser::Import_fromContext::Import_fromContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Import_fromContext::FROM() {
  return getToken(pyxasmParser::FROM, 0);
}

tree::TerminalNode* pyxasmParser::Import_fromContext::IMPORT() {
  return getToken(pyxasmParser::IMPORT, 0);
}

pyxasmParser::Dotted_nameContext* pyxasmParser::Import_fromContext::dotted_name() {
  return getRuleContext<pyxasmParser::Dotted_nameContext>(0);
}

tree::TerminalNode* pyxasmParser::Import_fromContext::STAR() {
  return getToken(pyxasmParser::STAR, 0);
}

tree::TerminalNode* pyxasmParser::Import_fromContext::OPEN_PAREN() {
  return getToken(pyxasmParser::OPEN_PAREN, 0);
}

pyxasmParser::Import_as_namesContext* pyxasmParser::Import_fromContext::import_as_names() {
  return getRuleContext<pyxasmParser::Import_as_namesContext>(0);
}

tree::TerminalNode* pyxasmParser::Import_fromContext::CLOSE_PAREN() {
  return getToken(pyxasmParser::CLOSE_PAREN, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Import_fromContext::DOT() {
  return getTokens(pyxasmParser::DOT);
}

tree::TerminalNode* pyxasmParser::Import_fromContext::DOT(size_t i) {
  return getToken(pyxasmParser::DOT, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Import_fromContext::ELLIPSIS() {
  return getTokens(pyxasmParser::ELLIPSIS);
}

tree::TerminalNode* pyxasmParser::Import_fromContext::ELLIPSIS(size_t i) {
  return getToken(pyxasmParser::ELLIPSIS, i);
}


size_t pyxasmParser::Import_fromContext::getRuleIndex() const {
  return pyxasmParser::RuleImport_from;
}

void pyxasmParser::Import_fromContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImport_from(this);
}

void pyxasmParser::Import_fromContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImport_from(this);
}


antlrcpp::Any pyxasmParser::Import_fromContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitImport_from(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Import_fromContext* pyxasmParser::import_from() {
  Import_fromContext *_localctx = _tracker.createInstance<Import_fromContext>(_ctx, getState());
  enterRule(_localctx, 60, pyxasmParser::RuleImport_from);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(519);
    match(pyxasmParser::FROM);
    setState(532);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 71, _ctx)) {
    case 1: {
      setState(523);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == pyxasmParser::DOT

      || _la == pyxasmParser::ELLIPSIS) {
        setState(520);
        _la = _input->LA(1);
        if (!(_la == pyxasmParser::DOT

        || _la == pyxasmParser::ELLIPSIS)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(525);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(526);
      dotted_name();
      break;
    }

    case 2: {
      setState(528); 
      _errHandler->sync(this);
      _la = _input->LA(1);
      do {
        setState(527);
        _la = _input->LA(1);
        if (!(_la == pyxasmParser::DOT

        || _la == pyxasmParser::ELLIPSIS)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(530); 
        _errHandler->sync(this);
        _la = _input->LA(1);
      } while (_la == pyxasmParser::DOT

      || _la == pyxasmParser::ELLIPSIS);
      break;
    }

    }
    setState(534);
    match(pyxasmParser::IMPORT);
    setState(541);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STAR: {
        setState(535);
        match(pyxasmParser::STAR);
        break;
      }

      case pyxasmParser::OPEN_PAREN: {
        setState(536);
        match(pyxasmParser::OPEN_PAREN);
        setState(537);
        import_as_names();
        setState(538);
        match(pyxasmParser::CLOSE_PAREN);
        break;
      }

      case pyxasmParser::NAME: {
        setState(540);
        import_as_names();
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

//----------------- Import_as_nameContext ------------------------------------------------------------------

pyxasmParser::Import_as_nameContext::Import_as_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> pyxasmParser::Import_as_nameContext::NAME() {
  return getTokens(pyxasmParser::NAME);
}

tree::TerminalNode* pyxasmParser::Import_as_nameContext::NAME(size_t i) {
  return getToken(pyxasmParser::NAME, i);
}

tree::TerminalNode* pyxasmParser::Import_as_nameContext::AS() {
  return getToken(pyxasmParser::AS, 0);
}


size_t pyxasmParser::Import_as_nameContext::getRuleIndex() const {
  return pyxasmParser::RuleImport_as_name;
}

void pyxasmParser::Import_as_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImport_as_name(this);
}

void pyxasmParser::Import_as_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImport_as_name(this);
}


antlrcpp::Any pyxasmParser::Import_as_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitImport_as_name(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Import_as_nameContext* pyxasmParser::import_as_name() {
  Import_as_nameContext *_localctx = _tracker.createInstance<Import_as_nameContext>(_ctx, getState());
  enterRule(_localctx, 62, pyxasmParser::RuleImport_as_name);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(543);
    match(pyxasmParser::NAME);
    setState(546);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::AS) {
      setState(544);
      match(pyxasmParser::AS);
      setState(545);
      match(pyxasmParser::NAME);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Dotted_as_nameContext ------------------------------------------------------------------

pyxasmParser::Dotted_as_nameContext::Dotted_as_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Dotted_nameContext* pyxasmParser::Dotted_as_nameContext::dotted_name() {
  return getRuleContext<pyxasmParser::Dotted_nameContext>(0);
}

tree::TerminalNode* pyxasmParser::Dotted_as_nameContext::AS() {
  return getToken(pyxasmParser::AS, 0);
}

tree::TerminalNode* pyxasmParser::Dotted_as_nameContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}


size_t pyxasmParser::Dotted_as_nameContext::getRuleIndex() const {
  return pyxasmParser::RuleDotted_as_name;
}

void pyxasmParser::Dotted_as_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDotted_as_name(this);
}

void pyxasmParser::Dotted_as_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDotted_as_name(this);
}


antlrcpp::Any pyxasmParser::Dotted_as_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDotted_as_name(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Dotted_as_nameContext* pyxasmParser::dotted_as_name() {
  Dotted_as_nameContext *_localctx = _tracker.createInstance<Dotted_as_nameContext>(_ctx, getState());
  enterRule(_localctx, 64, pyxasmParser::RuleDotted_as_name);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(548);
    dotted_name();
    setState(551);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::AS) {
      setState(549);
      match(pyxasmParser::AS);
      setState(550);
      match(pyxasmParser::NAME);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Import_as_namesContext ------------------------------------------------------------------

pyxasmParser::Import_as_namesContext::Import_as_namesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Import_as_nameContext *> pyxasmParser::Import_as_namesContext::import_as_name() {
  return getRuleContexts<pyxasmParser::Import_as_nameContext>();
}

pyxasmParser::Import_as_nameContext* pyxasmParser::Import_as_namesContext::import_as_name(size_t i) {
  return getRuleContext<pyxasmParser::Import_as_nameContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Import_as_namesContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::Import_as_namesContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::Import_as_namesContext::getRuleIndex() const {
  return pyxasmParser::RuleImport_as_names;
}

void pyxasmParser::Import_as_namesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImport_as_names(this);
}

void pyxasmParser::Import_as_namesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImport_as_names(this);
}


antlrcpp::Any pyxasmParser::Import_as_namesContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitImport_as_names(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Import_as_namesContext* pyxasmParser::import_as_names() {
  Import_as_namesContext *_localctx = _tracker.createInstance<Import_as_namesContext>(_ctx, getState());
  enterRule(_localctx, 66, pyxasmParser::RuleImport_as_names);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(553);
    import_as_name();
    setState(558);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 75, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(554);
        match(pyxasmParser::COMMA);
        setState(555);
        import_as_name(); 
      }
      setState(560);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 75, _ctx);
    }
    setState(562);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COMMA) {
      setState(561);
      match(pyxasmParser::COMMA);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Dotted_as_namesContext ------------------------------------------------------------------

pyxasmParser::Dotted_as_namesContext::Dotted_as_namesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Dotted_as_nameContext *> pyxasmParser::Dotted_as_namesContext::dotted_as_name() {
  return getRuleContexts<pyxasmParser::Dotted_as_nameContext>();
}

pyxasmParser::Dotted_as_nameContext* pyxasmParser::Dotted_as_namesContext::dotted_as_name(size_t i) {
  return getRuleContext<pyxasmParser::Dotted_as_nameContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Dotted_as_namesContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::Dotted_as_namesContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::Dotted_as_namesContext::getRuleIndex() const {
  return pyxasmParser::RuleDotted_as_names;
}

void pyxasmParser::Dotted_as_namesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDotted_as_names(this);
}

void pyxasmParser::Dotted_as_namesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDotted_as_names(this);
}


antlrcpp::Any pyxasmParser::Dotted_as_namesContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDotted_as_names(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Dotted_as_namesContext* pyxasmParser::dotted_as_names() {
  Dotted_as_namesContext *_localctx = _tracker.createInstance<Dotted_as_namesContext>(_ctx, getState());
  enterRule(_localctx, 68, pyxasmParser::RuleDotted_as_names);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(564);
    dotted_as_name();
    setState(569);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::COMMA) {
      setState(565);
      match(pyxasmParser::COMMA);
      setState(566);
      dotted_as_name();
      setState(571);
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

//----------------- Dotted_nameContext ------------------------------------------------------------------

pyxasmParser::Dotted_nameContext::Dotted_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> pyxasmParser::Dotted_nameContext::NAME() {
  return getTokens(pyxasmParser::NAME);
}

tree::TerminalNode* pyxasmParser::Dotted_nameContext::NAME(size_t i) {
  return getToken(pyxasmParser::NAME, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Dotted_nameContext::DOT() {
  return getTokens(pyxasmParser::DOT);
}

tree::TerminalNode* pyxasmParser::Dotted_nameContext::DOT(size_t i) {
  return getToken(pyxasmParser::DOT, i);
}


size_t pyxasmParser::Dotted_nameContext::getRuleIndex() const {
  return pyxasmParser::RuleDotted_name;
}

void pyxasmParser::Dotted_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDotted_name(this);
}

void pyxasmParser::Dotted_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDotted_name(this);
}


antlrcpp::Any pyxasmParser::Dotted_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDotted_name(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Dotted_nameContext* pyxasmParser::dotted_name() {
  Dotted_nameContext *_localctx = _tracker.createInstance<Dotted_nameContext>(_ctx, getState());
  enterRule(_localctx, 70, pyxasmParser::RuleDotted_name);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(572);
    match(pyxasmParser::NAME);
    setState(577);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::DOT) {
      setState(573);
      match(pyxasmParser::DOT);
      setState(574);
      match(pyxasmParser::NAME);
      setState(579);
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

//----------------- Global_stmtContext ------------------------------------------------------------------

pyxasmParser::Global_stmtContext::Global_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Global_stmtContext::GLOBAL() {
  return getToken(pyxasmParser::GLOBAL, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Global_stmtContext::NAME() {
  return getTokens(pyxasmParser::NAME);
}

tree::TerminalNode* pyxasmParser::Global_stmtContext::NAME(size_t i) {
  return getToken(pyxasmParser::NAME, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Global_stmtContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::Global_stmtContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::Global_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleGlobal_stmt;
}

void pyxasmParser::Global_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGlobal_stmt(this);
}

void pyxasmParser::Global_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGlobal_stmt(this);
}


antlrcpp::Any pyxasmParser::Global_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitGlobal_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Global_stmtContext* pyxasmParser::global_stmt() {
  Global_stmtContext *_localctx = _tracker.createInstance<Global_stmtContext>(_ctx, getState());
  enterRule(_localctx, 72, pyxasmParser::RuleGlobal_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(580);
    match(pyxasmParser::GLOBAL);
    setState(581);
    match(pyxasmParser::NAME);
    setState(586);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::COMMA) {
      setState(582);
      match(pyxasmParser::COMMA);
      setState(583);
      match(pyxasmParser::NAME);
      setState(588);
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

//----------------- Nonlocal_stmtContext ------------------------------------------------------------------

pyxasmParser::Nonlocal_stmtContext::Nonlocal_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Nonlocal_stmtContext::NONLOCAL() {
  return getToken(pyxasmParser::NONLOCAL, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Nonlocal_stmtContext::NAME() {
  return getTokens(pyxasmParser::NAME);
}

tree::TerminalNode* pyxasmParser::Nonlocal_stmtContext::NAME(size_t i) {
  return getToken(pyxasmParser::NAME, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Nonlocal_stmtContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::Nonlocal_stmtContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::Nonlocal_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleNonlocal_stmt;
}

void pyxasmParser::Nonlocal_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNonlocal_stmt(this);
}

void pyxasmParser::Nonlocal_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNonlocal_stmt(this);
}


antlrcpp::Any pyxasmParser::Nonlocal_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitNonlocal_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Nonlocal_stmtContext* pyxasmParser::nonlocal_stmt() {
  Nonlocal_stmtContext *_localctx = _tracker.createInstance<Nonlocal_stmtContext>(_ctx, getState());
  enterRule(_localctx, 74, pyxasmParser::RuleNonlocal_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(589);
    match(pyxasmParser::NONLOCAL);
    setState(590);
    match(pyxasmParser::NAME);
    setState(595);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::COMMA) {
      setState(591);
      match(pyxasmParser::COMMA);
      setState(592);
      match(pyxasmParser::NAME);
      setState(597);
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

//----------------- Assert_stmtContext ------------------------------------------------------------------

pyxasmParser::Assert_stmtContext::Assert_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Assert_stmtContext::ASSERT() {
  return getToken(pyxasmParser::ASSERT, 0);
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::Assert_stmtContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::Assert_stmtContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

tree::TerminalNode* pyxasmParser::Assert_stmtContext::COMMA() {
  return getToken(pyxasmParser::COMMA, 0);
}


size_t pyxasmParser::Assert_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleAssert_stmt;
}

void pyxasmParser::Assert_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssert_stmt(this);
}

void pyxasmParser::Assert_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssert_stmt(this);
}


antlrcpp::Any pyxasmParser::Assert_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAssert_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Assert_stmtContext* pyxasmParser::assert_stmt() {
  Assert_stmtContext *_localctx = _tracker.createInstance<Assert_stmtContext>(_ctx, getState());
  enterRule(_localctx, 76, pyxasmParser::RuleAssert_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(598);
    match(pyxasmParser::ASSERT);
    setState(599);
    test();
    setState(602);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COMMA) {
      setState(600);
      match(pyxasmParser::COMMA);
      setState(601);
      test();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Compound_stmtContext ------------------------------------------------------------------

pyxasmParser::Compound_stmtContext::Compound_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::If_stmtContext* pyxasmParser::Compound_stmtContext::if_stmt() {
  return getRuleContext<pyxasmParser::If_stmtContext>(0);
}

pyxasmParser::While_stmtContext* pyxasmParser::Compound_stmtContext::while_stmt() {
  return getRuleContext<pyxasmParser::While_stmtContext>(0);
}

pyxasmParser::For_stmtContext* pyxasmParser::Compound_stmtContext::for_stmt() {
  return getRuleContext<pyxasmParser::For_stmtContext>(0);
}

pyxasmParser::Try_stmtContext* pyxasmParser::Compound_stmtContext::try_stmt() {
  return getRuleContext<pyxasmParser::Try_stmtContext>(0);
}

pyxasmParser::With_stmtContext* pyxasmParser::Compound_stmtContext::with_stmt() {
  return getRuleContext<pyxasmParser::With_stmtContext>(0);
}

pyxasmParser::FuncdefContext* pyxasmParser::Compound_stmtContext::funcdef() {
  return getRuleContext<pyxasmParser::FuncdefContext>(0);
}

pyxasmParser::ClassdefContext* pyxasmParser::Compound_stmtContext::classdef() {
  return getRuleContext<pyxasmParser::ClassdefContext>(0);
}

pyxasmParser::DecoratedContext* pyxasmParser::Compound_stmtContext::decorated() {
  return getRuleContext<pyxasmParser::DecoratedContext>(0);
}

pyxasmParser::Async_stmtContext* pyxasmParser::Compound_stmtContext::async_stmt() {
  return getRuleContext<pyxasmParser::Async_stmtContext>(0);
}


size_t pyxasmParser::Compound_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleCompound_stmt;
}

void pyxasmParser::Compound_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompound_stmt(this);
}

void pyxasmParser::Compound_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompound_stmt(this);
}


antlrcpp::Any pyxasmParser::Compound_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitCompound_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Compound_stmtContext* pyxasmParser::compound_stmt() {
  Compound_stmtContext *_localctx = _tracker.createInstance<Compound_stmtContext>(_ctx, getState());
  enterRule(_localctx, 78, pyxasmParser::RuleCompound_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(613);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::IF: {
        enterOuterAlt(_localctx, 1);
        setState(604);
        if_stmt();
        break;
      }

      case pyxasmParser::WHILE: {
        enterOuterAlt(_localctx, 2);
        setState(605);
        while_stmt();
        break;
      }

      case pyxasmParser::FOR: {
        enterOuterAlt(_localctx, 3);
        setState(606);
        for_stmt();
        break;
      }

      case pyxasmParser::TRY: {
        enterOuterAlt(_localctx, 4);
        setState(607);
        try_stmt();
        break;
      }

      case pyxasmParser::WITH: {
        enterOuterAlt(_localctx, 5);
        setState(608);
        with_stmt();
        break;
      }

      case pyxasmParser::DEF: {
        enterOuterAlt(_localctx, 6);
        setState(609);
        funcdef();
        break;
      }

      case pyxasmParser::CLASS: {
        enterOuterAlt(_localctx, 7);
        setState(610);
        classdef();
        break;
      }

      case pyxasmParser::AT: {
        enterOuterAlt(_localctx, 8);
        setState(611);
        decorated();
        break;
      }

      case pyxasmParser::ASYNC: {
        enterOuterAlt(_localctx, 9);
        setState(612);
        async_stmt();
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

//----------------- Async_stmtContext ------------------------------------------------------------------

pyxasmParser::Async_stmtContext::Async_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Async_stmtContext::ASYNC() {
  return getToken(pyxasmParser::ASYNC, 0);
}

pyxasmParser::FuncdefContext* pyxasmParser::Async_stmtContext::funcdef() {
  return getRuleContext<pyxasmParser::FuncdefContext>(0);
}

pyxasmParser::With_stmtContext* pyxasmParser::Async_stmtContext::with_stmt() {
  return getRuleContext<pyxasmParser::With_stmtContext>(0);
}

pyxasmParser::For_stmtContext* pyxasmParser::Async_stmtContext::for_stmt() {
  return getRuleContext<pyxasmParser::For_stmtContext>(0);
}


size_t pyxasmParser::Async_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleAsync_stmt;
}

void pyxasmParser::Async_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAsync_stmt(this);
}

void pyxasmParser::Async_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAsync_stmt(this);
}


antlrcpp::Any pyxasmParser::Async_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAsync_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Async_stmtContext* pyxasmParser::async_stmt() {
  Async_stmtContext *_localctx = _tracker.createInstance<Async_stmtContext>(_ctx, getState());
  enterRule(_localctx, 80, pyxasmParser::RuleAsync_stmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(615);
    match(pyxasmParser::ASYNC);
    setState(619);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::DEF: {
        setState(616);
        funcdef();
        break;
      }

      case pyxasmParser::WITH: {
        setState(617);
        with_stmt();
        break;
      }

      case pyxasmParser::FOR: {
        setState(618);
        for_stmt();
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

//----------------- If_stmtContext ------------------------------------------------------------------

pyxasmParser::If_stmtContext::If_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::If_stmtContext::IF() {
  return getToken(pyxasmParser::IF, 0);
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::If_stmtContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::If_stmtContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::If_stmtContext::COLON() {
  return getTokens(pyxasmParser::COLON);
}

tree::TerminalNode* pyxasmParser::If_stmtContext::COLON(size_t i) {
  return getToken(pyxasmParser::COLON, i);
}

std::vector<pyxasmParser::SuiteContext *> pyxasmParser::If_stmtContext::suite() {
  return getRuleContexts<pyxasmParser::SuiteContext>();
}

pyxasmParser::SuiteContext* pyxasmParser::If_stmtContext::suite(size_t i) {
  return getRuleContext<pyxasmParser::SuiteContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::If_stmtContext::ELIF() {
  return getTokens(pyxasmParser::ELIF);
}

tree::TerminalNode* pyxasmParser::If_stmtContext::ELIF(size_t i) {
  return getToken(pyxasmParser::ELIF, i);
}

tree::TerminalNode* pyxasmParser::If_stmtContext::ELSE() {
  return getToken(pyxasmParser::ELSE, 0);
}


size_t pyxasmParser::If_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleIf_stmt;
}

void pyxasmParser::If_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIf_stmt(this);
}

void pyxasmParser::If_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIf_stmt(this);
}


antlrcpp::Any pyxasmParser::If_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitIf_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::If_stmtContext* pyxasmParser::if_stmt() {
  If_stmtContext *_localctx = _tracker.createInstance<If_stmtContext>(_ctx, getState());
  enterRule(_localctx, 82, pyxasmParser::RuleIf_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(621);
    match(pyxasmParser::IF);
    setState(622);
    test();
    setState(623);
    match(pyxasmParser::COLON);
    setState(624);
    suite();
    setState(632);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::ELIF) {
      setState(625);
      match(pyxasmParser::ELIF);
      setState(626);
      test();
      setState(627);
      match(pyxasmParser::COLON);
      setState(628);
      suite();
      setState(634);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(638);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::ELSE) {
      setState(635);
      match(pyxasmParser::ELSE);
      setState(636);
      match(pyxasmParser::COLON);
      setState(637);
      suite();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- While_stmtContext ------------------------------------------------------------------

pyxasmParser::While_stmtContext::While_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::While_stmtContext::WHILE() {
  return getToken(pyxasmParser::WHILE, 0);
}

pyxasmParser::TestContext* pyxasmParser::While_stmtContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}

std::vector<tree::TerminalNode *> pyxasmParser::While_stmtContext::COLON() {
  return getTokens(pyxasmParser::COLON);
}

tree::TerminalNode* pyxasmParser::While_stmtContext::COLON(size_t i) {
  return getToken(pyxasmParser::COLON, i);
}

std::vector<pyxasmParser::SuiteContext *> pyxasmParser::While_stmtContext::suite() {
  return getRuleContexts<pyxasmParser::SuiteContext>();
}

pyxasmParser::SuiteContext* pyxasmParser::While_stmtContext::suite(size_t i) {
  return getRuleContext<pyxasmParser::SuiteContext>(i);
}

tree::TerminalNode* pyxasmParser::While_stmtContext::ELSE() {
  return getToken(pyxasmParser::ELSE, 0);
}


size_t pyxasmParser::While_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleWhile_stmt;
}

void pyxasmParser::While_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhile_stmt(this);
}

void pyxasmParser::While_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhile_stmt(this);
}


antlrcpp::Any pyxasmParser::While_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitWhile_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::While_stmtContext* pyxasmParser::while_stmt() {
  While_stmtContext *_localctx = _tracker.createInstance<While_stmtContext>(_ctx, getState());
  enterRule(_localctx, 84, pyxasmParser::RuleWhile_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(640);
    match(pyxasmParser::WHILE);
    setState(641);
    test();
    setState(642);
    match(pyxasmParser::COLON);
    setState(643);
    suite();
    setState(647);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::ELSE) {
      setState(644);
      match(pyxasmParser::ELSE);
      setState(645);
      match(pyxasmParser::COLON);
      setState(646);
      suite();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- For_stmtContext ------------------------------------------------------------------

pyxasmParser::For_stmtContext::For_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::For_stmtContext::FOR() {
  return getToken(pyxasmParser::FOR, 0);
}

pyxasmParser::ExprlistContext* pyxasmParser::For_stmtContext::exprlist() {
  return getRuleContext<pyxasmParser::ExprlistContext>(0);
}

tree::TerminalNode* pyxasmParser::For_stmtContext::IN() {
  return getToken(pyxasmParser::IN, 0);
}

pyxasmParser::TestlistContext* pyxasmParser::For_stmtContext::testlist() {
  return getRuleContext<pyxasmParser::TestlistContext>(0);
}

std::vector<tree::TerminalNode *> pyxasmParser::For_stmtContext::COLON() {
  return getTokens(pyxasmParser::COLON);
}

tree::TerminalNode* pyxasmParser::For_stmtContext::COLON(size_t i) {
  return getToken(pyxasmParser::COLON, i);
}

std::vector<pyxasmParser::SuiteContext *> pyxasmParser::For_stmtContext::suite() {
  return getRuleContexts<pyxasmParser::SuiteContext>();
}

pyxasmParser::SuiteContext* pyxasmParser::For_stmtContext::suite(size_t i) {
  return getRuleContext<pyxasmParser::SuiteContext>(i);
}

tree::TerminalNode* pyxasmParser::For_stmtContext::ELSE() {
  return getToken(pyxasmParser::ELSE, 0);
}


size_t pyxasmParser::For_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleFor_stmt;
}

void pyxasmParser::For_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFor_stmt(this);
}

void pyxasmParser::For_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFor_stmt(this);
}


antlrcpp::Any pyxasmParser::For_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitFor_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::For_stmtContext* pyxasmParser::for_stmt() {
  For_stmtContext *_localctx = _tracker.createInstance<For_stmtContext>(_ctx, getState());
  enterRule(_localctx, 86, pyxasmParser::RuleFor_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(649);
    match(pyxasmParser::FOR);
    setState(650);
    exprlist();
    setState(651);
    match(pyxasmParser::IN);
    setState(652);
    testlist();
    setState(653);
    match(pyxasmParser::COLON);
    setState(654);
    suite();
    setState(658);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::ELSE) {
      setState(655);
      match(pyxasmParser::ELSE);
      setState(656);
      match(pyxasmParser::COLON);
      setState(657);
      suite();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Try_stmtContext ------------------------------------------------------------------

pyxasmParser::Try_stmtContext::Try_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Try_stmtContext::TRY() {
  return getToken(pyxasmParser::TRY, 0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Try_stmtContext::COLON() {
  return getTokens(pyxasmParser::COLON);
}

tree::TerminalNode* pyxasmParser::Try_stmtContext::COLON(size_t i) {
  return getToken(pyxasmParser::COLON, i);
}

std::vector<pyxasmParser::SuiteContext *> pyxasmParser::Try_stmtContext::suite() {
  return getRuleContexts<pyxasmParser::SuiteContext>();
}

pyxasmParser::SuiteContext* pyxasmParser::Try_stmtContext::suite(size_t i) {
  return getRuleContext<pyxasmParser::SuiteContext>(i);
}

tree::TerminalNode* pyxasmParser::Try_stmtContext::FINALLY() {
  return getToken(pyxasmParser::FINALLY, 0);
}

std::vector<pyxasmParser::Except_clauseContext *> pyxasmParser::Try_stmtContext::except_clause() {
  return getRuleContexts<pyxasmParser::Except_clauseContext>();
}

pyxasmParser::Except_clauseContext* pyxasmParser::Try_stmtContext::except_clause(size_t i) {
  return getRuleContext<pyxasmParser::Except_clauseContext>(i);
}

tree::TerminalNode* pyxasmParser::Try_stmtContext::ELSE() {
  return getToken(pyxasmParser::ELSE, 0);
}


size_t pyxasmParser::Try_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleTry_stmt;
}

void pyxasmParser::Try_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTry_stmt(this);
}

void pyxasmParser::Try_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTry_stmt(this);
}


antlrcpp::Any pyxasmParser::Try_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTry_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Try_stmtContext* pyxasmParser::try_stmt() {
  Try_stmtContext *_localctx = _tracker.createInstance<Try_stmtContext>(_ctx, getState());
  enterRule(_localctx, 88, pyxasmParser::RuleTry_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(660);
    match(pyxasmParser::TRY);
    setState(661);
    match(pyxasmParser::COLON);
    setState(662);
    suite();
    setState(684);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::EXCEPT: {
        setState(667); 
        _errHandler->sync(this);
        _la = _input->LA(1);
        do {
          setState(663);
          except_clause();
          setState(664);
          match(pyxasmParser::COLON);
          setState(665);
          suite();
          setState(669); 
          _errHandler->sync(this);
          _la = _input->LA(1);
        } while (_la == pyxasmParser::EXCEPT);
        setState(674);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::ELSE) {
          setState(671);
          match(pyxasmParser::ELSE);
          setState(672);
          match(pyxasmParser::COLON);
          setState(673);
          suite();
        }
        setState(679);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::FINALLY) {
          setState(676);
          match(pyxasmParser::FINALLY);
          setState(677);
          match(pyxasmParser::COLON);
          setState(678);
          suite();
        }
        break;
      }

      case pyxasmParser::FINALLY: {
        setState(681);
        match(pyxasmParser::FINALLY);
        setState(682);
        match(pyxasmParser::COLON);
        setState(683);
        suite();
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

//----------------- With_stmtContext ------------------------------------------------------------------

pyxasmParser::With_stmtContext::With_stmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::With_stmtContext::WITH() {
  return getToken(pyxasmParser::WITH, 0);
}

std::vector<pyxasmParser::With_itemContext *> pyxasmParser::With_stmtContext::with_item() {
  return getRuleContexts<pyxasmParser::With_itemContext>();
}

pyxasmParser::With_itemContext* pyxasmParser::With_stmtContext::with_item(size_t i) {
  return getRuleContext<pyxasmParser::With_itemContext>(i);
}

tree::TerminalNode* pyxasmParser::With_stmtContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::SuiteContext* pyxasmParser::With_stmtContext::suite() {
  return getRuleContext<pyxasmParser::SuiteContext>(0);
}

std::vector<tree::TerminalNode *> pyxasmParser::With_stmtContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::With_stmtContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::With_stmtContext::getRuleIndex() const {
  return pyxasmParser::RuleWith_stmt;
}

void pyxasmParser::With_stmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWith_stmt(this);
}

void pyxasmParser::With_stmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWith_stmt(this);
}


antlrcpp::Any pyxasmParser::With_stmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitWith_stmt(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::With_stmtContext* pyxasmParser::with_stmt() {
  With_stmtContext *_localctx = _tracker.createInstance<With_stmtContext>(_ctx, getState());
  enterRule(_localctx, 90, pyxasmParser::RuleWith_stmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(686);
    match(pyxasmParser::WITH);
    setState(687);
    with_item();
    setState(692);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::COMMA) {
      setState(688);
      match(pyxasmParser::COMMA);
      setState(689);
      with_item();
      setState(694);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(695);
    match(pyxasmParser::COLON);
    setState(696);
    suite();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- With_itemContext ------------------------------------------------------------------

pyxasmParser::With_itemContext::With_itemContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::TestContext* pyxasmParser::With_itemContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}

tree::TerminalNode* pyxasmParser::With_itemContext::AS() {
  return getToken(pyxasmParser::AS, 0);
}

pyxasmParser::ExprContext* pyxasmParser::With_itemContext::expr() {
  return getRuleContext<pyxasmParser::ExprContext>(0);
}


size_t pyxasmParser::With_itemContext::getRuleIndex() const {
  return pyxasmParser::RuleWith_item;
}

void pyxasmParser::With_itemContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWith_item(this);
}

void pyxasmParser::With_itemContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWith_item(this);
}


antlrcpp::Any pyxasmParser::With_itemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitWith_item(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::With_itemContext* pyxasmParser::with_item() {
  With_itemContext *_localctx = _tracker.createInstance<With_itemContext>(_ctx, getState());
  enterRule(_localctx, 92, pyxasmParser::RuleWith_item);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(698);
    test();
    setState(701);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::AS) {
      setState(699);
      match(pyxasmParser::AS);
      setState(700);
      expr();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Except_clauseContext ------------------------------------------------------------------

pyxasmParser::Except_clauseContext::Except_clauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Except_clauseContext::EXCEPT() {
  return getToken(pyxasmParser::EXCEPT, 0);
}

pyxasmParser::TestContext* pyxasmParser::Except_clauseContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}

tree::TerminalNode* pyxasmParser::Except_clauseContext::AS() {
  return getToken(pyxasmParser::AS, 0);
}

tree::TerminalNode* pyxasmParser::Except_clauseContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}


size_t pyxasmParser::Except_clauseContext::getRuleIndex() const {
  return pyxasmParser::RuleExcept_clause;
}

void pyxasmParser::Except_clauseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExcept_clause(this);
}

void pyxasmParser::Except_clauseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExcept_clause(this);
}


antlrcpp::Any pyxasmParser::Except_clauseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitExcept_clause(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Except_clauseContext* pyxasmParser::except_clause() {
  Except_clauseContext *_localctx = _tracker.createInstance<Except_clauseContext>(_ctx, getState());
  enterRule(_localctx, 94, pyxasmParser::RuleExcept_clause);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(703);
    match(pyxasmParser::EXCEPT);
    setState(709);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
      | (1ULL << pyxasmParser::NUMBER)
      | (1ULL << pyxasmParser::LAMBDA)
      | (1ULL << pyxasmParser::NOT)
      | (1ULL << pyxasmParser::NONE)
      | (1ULL << pyxasmParser::TRUE)
      | (1ULL << pyxasmParser::FALSE)
      | (1ULL << pyxasmParser::AWAIT)
      | (1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::ELLIPSIS)
      | (1ULL << pyxasmParser::OPEN_PAREN)
      | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
      | (1ULL << (pyxasmParser::MINUS - 66))
      | (1ULL << (pyxasmParser::NOT_OP - 66))
      | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
      setState(704);
      test();
      setState(707);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == pyxasmParser::AS) {
        setState(705);
        match(pyxasmParser::AS);
        setState(706);
        match(pyxasmParser::NAME);
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

//----------------- SuiteContext ------------------------------------------------------------------

pyxasmParser::SuiteContext::SuiteContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Simple_stmtContext* pyxasmParser::SuiteContext::simple_stmt() {
  return getRuleContext<pyxasmParser::Simple_stmtContext>(0);
}

tree::TerminalNode* pyxasmParser::SuiteContext::NEWLINE() {
  return getToken(pyxasmParser::NEWLINE, 0);
}

tree::TerminalNode* pyxasmParser::SuiteContext::INDENT() {
  return getToken(pyxasmParser::INDENT, 0);
}

tree::TerminalNode* pyxasmParser::SuiteContext::DEDENT() {
  return getToken(pyxasmParser::DEDENT, 0);
}

std::vector<pyxasmParser::StmtContext *> pyxasmParser::SuiteContext::stmt() {
  return getRuleContexts<pyxasmParser::StmtContext>();
}

pyxasmParser::StmtContext* pyxasmParser::SuiteContext::stmt(size_t i) {
  return getRuleContext<pyxasmParser::StmtContext>(i);
}


size_t pyxasmParser::SuiteContext::getRuleIndex() const {
  return pyxasmParser::RuleSuite;
}

void pyxasmParser::SuiteContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSuite(this);
}

void pyxasmParser::SuiteContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSuite(this);
}


antlrcpp::Any pyxasmParser::SuiteContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitSuite(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::SuiteContext* pyxasmParser::suite() {
  SuiteContext *_localctx = _tracker.createInstance<SuiteContext>(_ctx, getState());
  enterRule(_localctx, 96, pyxasmParser::RuleSuite);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(721);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::RETURN:
      case pyxasmParser::RAISE:
      case pyxasmParser::FROM:
      case pyxasmParser::IMPORT:
      case pyxasmParser::GLOBAL:
      case pyxasmParser::NONLOCAL:
      case pyxasmParser::ASSERT:
      case pyxasmParser::LAMBDA:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::YIELD:
      case pyxasmParser::DEL:
      case pyxasmParser::PASS:
      case pyxasmParser::CONTINUE:
      case pyxasmParser::BREAK:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::STAR:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 1);
        setState(711);
        simple_stmt();
        break;
      }

      case pyxasmParser::NEWLINE: {
        enterOuterAlt(_localctx, 2);
        setState(712);
        match(pyxasmParser::NEWLINE);
        setState(713);
        match(pyxasmParser::INDENT);
        setState(715); 
        _errHandler->sync(this);
        _la = _input->LA(1);
        do {
          setState(714);
          stmt();
          setState(717); 
          _errHandler->sync(this);
          _la = _input->LA(1);
        } while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
          | (1ULL << pyxasmParser::NUMBER)
          | (1ULL << pyxasmParser::DEF)
          | (1ULL << pyxasmParser::RETURN)
          | (1ULL << pyxasmParser::RAISE)
          | (1ULL << pyxasmParser::FROM)
          | (1ULL << pyxasmParser::IMPORT)
          | (1ULL << pyxasmParser::GLOBAL)
          | (1ULL << pyxasmParser::NONLOCAL)
          | (1ULL << pyxasmParser::ASSERT)
          | (1ULL << pyxasmParser::IF)
          | (1ULL << pyxasmParser::WHILE)
          | (1ULL << pyxasmParser::FOR)
          | (1ULL << pyxasmParser::TRY)
          | (1ULL << pyxasmParser::WITH)
          | (1ULL << pyxasmParser::LAMBDA)
          | (1ULL << pyxasmParser::NOT)
          | (1ULL << pyxasmParser::NONE)
          | (1ULL << pyxasmParser::TRUE)
          | (1ULL << pyxasmParser::FALSE)
          | (1ULL << pyxasmParser::CLASS)
          | (1ULL << pyxasmParser::YIELD)
          | (1ULL << pyxasmParser::DEL)
          | (1ULL << pyxasmParser::PASS)
          | (1ULL << pyxasmParser::CONTINUE)
          | (1ULL << pyxasmParser::BREAK)
          | (1ULL << pyxasmParser::ASYNC)
          | (1ULL << pyxasmParser::AWAIT)
          | (1ULL << pyxasmParser::NAME)
          | (1ULL << pyxasmParser::ELLIPSIS)
          | (1ULL << pyxasmParser::STAR)
          | (1ULL << pyxasmParser::OPEN_PAREN)
          | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
          | (1ULL << (pyxasmParser::MINUS - 66))
          | (1ULL << (pyxasmParser::NOT_OP - 66))
          | (1ULL << (pyxasmParser::OPEN_BRACE - 66))
          | (1ULL << (pyxasmParser::AT - 66)))) != 0));
        setState(719);
        match(pyxasmParser::DEDENT);
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

//----------------- TestContext ------------------------------------------------------------------

pyxasmParser::TestContext::TestContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Or_testContext *> pyxasmParser::TestContext::or_test() {
  return getRuleContexts<pyxasmParser::Or_testContext>();
}

pyxasmParser::Or_testContext* pyxasmParser::TestContext::or_test(size_t i) {
  return getRuleContext<pyxasmParser::Or_testContext>(i);
}

tree::TerminalNode* pyxasmParser::TestContext::IF() {
  return getToken(pyxasmParser::IF, 0);
}

tree::TerminalNode* pyxasmParser::TestContext::ELSE() {
  return getToken(pyxasmParser::ELSE, 0);
}

pyxasmParser::TestContext* pyxasmParser::TestContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}

pyxasmParser::LambdefContext* pyxasmParser::TestContext::lambdef() {
  return getRuleContext<pyxasmParser::LambdefContext>(0);
}


size_t pyxasmParser::TestContext::getRuleIndex() const {
  return pyxasmParser::RuleTest;
}

void pyxasmParser::TestContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTest(this);
}

void pyxasmParser::TestContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTest(this);
}


antlrcpp::Any pyxasmParser::TestContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTest(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::TestContext* pyxasmParser::test() {
  TestContext *_localctx = _tracker.createInstance<TestContext>(_ctx, getState());
  enterRule(_localctx, 98, pyxasmParser::RuleTest);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(732);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 1);
        setState(723);
        or_test();
        setState(729);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::IF) {
          setState(724);
          match(pyxasmParser::IF);
          setState(725);
          or_test();
          setState(726);
          match(pyxasmParser::ELSE);
          setState(727);
          test();
        }
        break;
      }

      case pyxasmParser::LAMBDA: {
        enterOuterAlt(_localctx, 2);
        setState(731);
        lambdef();
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

//----------------- Test_nocondContext ------------------------------------------------------------------

pyxasmParser::Test_nocondContext::Test_nocondContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Or_testContext* pyxasmParser::Test_nocondContext::or_test() {
  return getRuleContext<pyxasmParser::Or_testContext>(0);
}

pyxasmParser::Lambdef_nocondContext* pyxasmParser::Test_nocondContext::lambdef_nocond() {
  return getRuleContext<pyxasmParser::Lambdef_nocondContext>(0);
}


size_t pyxasmParser::Test_nocondContext::getRuleIndex() const {
  return pyxasmParser::RuleTest_nocond;
}

void pyxasmParser::Test_nocondContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTest_nocond(this);
}

void pyxasmParser::Test_nocondContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTest_nocond(this);
}


antlrcpp::Any pyxasmParser::Test_nocondContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTest_nocond(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Test_nocondContext* pyxasmParser::test_nocond() {
  Test_nocondContext *_localctx = _tracker.createInstance<Test_nocondContext>(_ctx, getState());
  enterRule(_localctx, 100, pyxasmParser::RuleTest_nocond);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(736);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 1);
        setState(734);
        or_test();
        break;
      }

      case pyxasmParser::LAMBDA: {
        enterOuterAlt(_localctx, 2);
        setState(735);
        lambdef_nocond();
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

//----------------- LambdefContext ------------------------------------------------------------------

pyxasmParser::LambdefContext::LambdefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::LambdefContext::LAMBDA() {
  return getToken(pyxasmParser::LAMBDA, 0);
}

tree::TerminalNode* pyxasmParser::LambdefContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::TestContext* pyxasmParser::LambdefContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}

pyxasmParser::VarargslistContext* pyxasmParser::LambdefContext::varargslist() {
  return getRuleContext<pyxasmParser::VarargslistContext>(0);
}


size_t pyxasmParser::LambdefContext::getRuleIndex() const {
  return pyxasmParser::RuleLambdef;
}

void pyxasmParser::LambdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLambdef(this);
}

void pyxasmParser::LambdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLambdef(this);
}


antlrcpp::Any pyxasmParser::LambdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitLambdef(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::LambdefContext* pyxasmParser::lambdef() {
  LambdefContext *_localctx = _tracker.createInstance<LambdefContext>(_ctx, getState());
  enterRule(_localctx, 102, pyxasmParser::RuleLambdef);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(738);
    match(pyxasmParser::LAMBDA);
    setState(740);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::STAR)
      | (1ULL << pyxasmParser::POWER))) != 0)) {
      setState(739);
      varargslist();
    }
    setState(742);
    match(pyxasmParser::COLON);
    setState(743);
    test();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Lambdef_nocondContext ------------------------------------------------------------------

pyxasmParser::Lambdef_nocondContext::Lambdef_nocondContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Lambdef_nocondContext::LAMBDA() {
  return getToken(pyxasmParser::LAMBDA, 0);
}

tree::TerminalNode* pyxasmParser::Lambdef_nocondContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::Test_nocondContext* pyxasmParser::Lambdef_nocondContext::test_nocond() {
  return getRuleContext<pyxasmParser::Test_nocondContext>(0);
}

pyxasmParser::VarargslistContext* pyxasmParser::Lambdef_nocondContext::varargslist() {
  return getRuleContext<pyxasmParser::VarargslistContext>(0);
}


size_t pyxasmParser::Lambdef_nocondContext::getRuleIndex() const {
  return pyxasmParser::RuleLambdef_nocond;
}

void pyxasmParser::Lambdef_nocondContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLambdef_nocond(this);
}

void pyxasmParser::Lambdef_nocondContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLambdef_nocond(this);
}


antlrcpp::Any pyxasmParser::Lambdef_nocondContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitLambdef_nocond(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Lambdef_nocondContext* pyxasmParser::lambdef_nocond() {
  Lambdef_nocondContext *_localctx = _tracker.createInstance<Lambdef_nocondContext>(_ctx, getState());
  enterRule(_localctx, 104, pyxasmParser::RuleLambdef_nocond);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(745);
    match(pyxasmParser::LAMBDA);
    setState(747);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::STAR)
      | (1ULL << pyxasmParser::POWER))) != 0)) {
      setState(746);
      varargslist();
    }
    setState(749);
    match(pyxasmParser::COLON);
    setState(750);
    test_nocond();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Or_testContext ------------------------------------------------------------------

pyxasmParser::Or_testContext::Or_testContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::And_testContext *> pyxasmParser::Or_testContext::and_test() {
  return getRuleContexts<pyxasmParser::And_testContext>();
}

pyxasmParser::And_testContext* pyxasmParser::Or_testContext::and_test(size_t i) {
  return getRuleContext<pyxasmParser::And_testContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Or_testContext::OR() {
  return getTokens(pyxasmParser::OR);
}

tree::TerminalNode* pyxasmParser::Or_testContext::OR(size_t i) {
  return getToken(pyxasmParser::OR, i);
}


size_t pyxasmParser::Or_testContext::getRuleIndex() const {
  return pyxasmParser::RuleOr_test;
}

void pyxasmParser::Or_testContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOr_test(this);
}

void pyxasmParser::Or_testContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOr_test(this);
}


antlrcpp::Any pyxasmParser::Or_testContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitOr_test(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Or_testContext* pyxasmParser::or_test() {
  Or_testContext *_localctx = _tracker.createInstance<Or_testContext>(_ctx, getState());
  enterRule(_localctx, 106, pyxasmParser::RuleOr_test);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(752);
    and_test();
    setState(757);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::OR) {
      setState(753);
      match(pyxasmParser::OR);
      setState(754);
      and_test();
      setState(759);
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

//----------------- And_testContext ------------------------------------------------------------------

pyxasmParser::And_testContext::And_testContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Not_testContext *> pyxasmParser::And_testContext::not_test() {
  return getRuleContexts<pyxasmParser::Not_testContext>();
}

pyxasmParser::Not_testContext* pyxasmParser::And_testContext::not_test(size_t i) {
  return getRuleContext<pyxasmParser::Not_testContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::And_testContext::AND() {
  return getTokens(pyxasmParser::AND);
}

tree::TerminalNode* pyxasmParser::And_testContext::AND(size_t i) {
  return getToken(pyxasmParser::AND, i);
}


size_t pyxasmParser::And_testContext::getRuleIndex() const {
  return pyxasmParser::RuleAnd_test;
}

void pyxasmParser::And_testContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnd_test(this);
}

void pyxasmParser::And_testContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnd_test(this);
}


antlrcpp::Any pyxasmParser::And_testContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAnd_test(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::And_testContext* pyxasmParser::and_test() {
  And_testContext *_localctx = _tracker.createInstance<And_testContext>(_ctx, getState());
  enterRule(_localctx, 108, pyxasmParser::RuleAnd_test);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(760);
    not_test();
    setState(765);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::AND) {
      setState(761);
      match(pyxasmParser::AND);
      setState(762);
      not_test();
      setState(767);
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

//----------------- Not_testContext ------------------------------------------------------------------

pyxasmParser::Not_testContext::Not_testContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Not_testContext::NOT() {
  return getToken(pyxasmParser::NOT, 0);
}

pyxasmParser::Not_testContext* pyxasmParser::Not_testContext::not_test() {
  return getRuleContext<pyxasmParser::Not_testContext>(0);
}

pyxasmParser::ComparisonContext* pyxasmParser::Not_testContext::comparison() {
  return getRuleContext<pyxasmParser::ComparisonContext>(0);
}


size_t pyxasmParser::Not_testContext::getRuleIndex() const {
  return pyxasmParser::RuleNot_test;
}

void pyxasmParser::Not_testContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNot_test(this);
}

void pyxasmParser::Not_testContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNot_test(this);
}


antlrcpp::Any pyxasmParser::Not_testContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitNot_test(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Not_testContext* pyxasmParser::not_test() {
  Not_testContext *_localctx = _tracker.createInstance<Not_testContext>(_ctx, getState());
  enterRule(_localctx, 110, pyxasmParser::RuleNot_test);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(771);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::NOT: {
        enterOuterAlt(_localctx, 1);
        setState(768);
        match(pyxasmParser::NOT);
        setState(769);
        not_test();
        break;
      }

      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 2);
        setState(770);
        comparison();
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

//----------------- ComparisonContext ------------------------------------------------------------------

pyxasmParser::ComparisonContext::ComparisonContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::ExprContext *> pyxasmParser::ComparisonContext::expr() {
  return getRuleContexts<pyxasmParser::ExprContext>();
}

pyxasmParser::ExprContext* pyxasmParser::ComparisonContext::expr(size_t i) {
  return getRuleContext<pyxasmParser::ExprContext>(i);
}

std::vector<pyxasmParser::Comp_opContext *> pyxasmParser::ComparisonContext::comp_op() {
  return getRuleContexts<pyxasmParser::Comp_opContext>();
}

pyxasmParser::Comp_opContext* pyxasmParser::ComparisonContext::comp_op(size_t i) {
  return getRuleContext<pyxasmParser::Comp_opContext>(i);
}


size_t pyxasmParser::ComparisonContext::getRuleIndex() const {
  return pyxasmParser::RuleComparison;
}

void pyxasmParser::ComparisonContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComparison(this);
}

void pyxasmParser::ComparisonContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComparison(this);
}


antlrcpp::Any pyxasmParser::ComparisonContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitComparison(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::ComparisonContext* pyxasmParser::comparison() {
  ComparisonContext *_localctx = _tracker.createInstance<ComparisonContext>(_ctx, getState());
  enterRule(_localctx, 112, pyxasmParser::RuleComparison);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(773);
    expr();
    setState(779);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (((((_la - 18) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 18)) & ((1ULL << (pyxasmParser::IN - 18))
      | (1ULL << (pyxasmParser::NOT - 18))
      | (1ULL << (pyxasmParser::IS - 18))
      | (1ULL << (pyxasmParser::LESS_THAN - 18))
      | (1ULL << (pyxasmParser::GREATER_THAN - 18))
      | (1ULL << (pyxasmParser::EQUALS - 18))
      | (1ULL << (pyxasmParser::GT_EQ - 18))
      | (1ULL << (pyxasmParser::LT_EQ - 18))
      | (1ULL << (pyxasmParser::NOT_EQ_1 - 18))
      | (1ULL << (pyxasmParser::NOT_EQ_2 - 18)))) != 0)) {
      setState(774);
      comp_op();
      setState(775);
      expr();
      setState(781);
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

//----------------- Comp_opContext ------------------------------------------------------------------

pyxasmParser::Comp_opContext::Comp_opContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Comp_opContext::LESS_THAN() {
  return getToken(pyxasmParser::LESS_THAN, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::GREATER_THAN() {
  return getToken(pyxasmParser::GREATER_THAN, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::EQUALS() {
  return getToken(pyxasmParser::EQUALS, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::GT_EQ() {
  return getToken(pyxasmParser::GT_EQ, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::LT_EQ() {
  return getToken(pyxasmParser::LT_EQ, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::NOT_EQ_1() {
  return getToken(pyxasmParser::NOT_EQ_1, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::NOT_EQ_2() {
  return getToken(pyxasmParser::NOT_EQ_2, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::IN() {
  return getToken(pyxasmParser::IN, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::NOT() {
  return getToken(pyxasmParser::NOT, 0);
}

tree::TerminalNode* pyxasmParser::Comp_opContext::IS() {
  return getToken(pyxasmParser::IS, 0);
}


size_t pyxasmParser::Comp_opContext::getRuleIndex() const {
  return pyxasmParser::RuleComp_op;
}

void pyxasmParser::Comp_opContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComp_op(this);
}

void pyxasmParser::Comp_opContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComp_op(this);
}


antlrcpp::Any pyxasmParser::Comp_opContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitComp_op(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Comp_opContext* pyxasmParser::comp_op() {
  Comp_opContext *_localctx = _tracker.createInstance<Comp_opContext>(_ctx, getState());
  enterRule(_localctx, 114, pyxasmParser::RuleComp_op);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(795);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 107, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(782);
      match(pyxasmParser::LESS_THAN);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(783);
      match(pyxasmParser::GREATER_THAN);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(784);
      match(pyxasmParser::EQUALS);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(785);
      match(pyxasmParser::GT_EQ);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(786);
      match(pyxasmParser::LT_EQ);
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(787);
      match(pyxasmParser::NOT_EQ_1);
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(788);
      match(pyxasmParser::NOT_EQ_2);
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(789);
      match(pyxasmParser::IN);
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(790);
      match(pyxasmParser::NOT);
      setState(791);
      match(pyxasmParser::IN);
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(792);
      match(pyxasmParser::IS);
      break;
    }

    case 11: {
      enterOuterAlt(_localctx, 11);
      setState(793);
      match(pyxasmParser::IS);
      setState(794);
      match(pyxasmParser::NOT);
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

//----------------- Star_exprContext ------------------------------------------------------------------

pyxasmParser::Star_exprContext::Star_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Star_exprContext::STAR() {
  return getToken(pyxasmParser::STAR, 0);
}

pyxasmParser::ExprContext* pyxasmParser::Star_exprContext::expr() {
  return getRuleContext<pyxasmParser::ExprContext>(0);
}


size_t pyxasmParser::Star_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleStar_expr;
}

void pyxasmParser::Star_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStar_expr(this);
}

void pyxasmParser::Star_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStar_expr(this);
}


antlrcpp::Any pyxasmParser::Star_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitStar_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Star_exprContext* pyxasmParser::star_expr() {
  Star_exprContext *_localctx = _tracker.createInstance<Star_exprContext>(_ctx, getState());
  enterRule(_localctx, 116, pyxasmParser::RuleStar_expr);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(797);
    match(pyxasmParser::STAR);
    setState(798);
    expr();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprContext ------------------------------------------------------------------

pyxasmParser::ExprContext::ExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Xor_exprContext *> pyxasmParser::ExprContext::xor_expr() {
  return getRuleContexts<pyxasmParser::Xor_exprContext>();
}

pyxasmParser::Xor_exprContext* pyxasmParser::ExprContext::xor_expr(size_t i) {
  return getRuleContext<pyxasmParser::Xor_exprContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::ExprContext::OR_OP() {
  return getTokens(pyxasmParser::OR_OP);
}

tree::TerminalNode* pyxasmParser::ExprContext::OR_OP(size_t i) {
  return getToken(pyxasmParser::OR_OP, i);
}


size_t pyxasmParser::ExprContext::getRuleIndex() const {
  return pyxasmParser::RuleExpr;
}

void pyxasmParser::ExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpr(this);
}

void pyxasmParser::ExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpr(this);
}


antlrcpp::Any pyxasmParser::ExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitExpr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::ExprContext* pyxasmParser::expr() {
  ExprContext *_localctx = _tracker.createInstance<ExprContext>(_ctx, getState());
  enterRule(_localctx, 118, pyxasmParser::RuleExpr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(800);
    xor_expr();
    setState(805);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::OR_OP) {
      setState(801);
      match(pyxasmParser::OR_OP);
      setState(802);
      xor_expr();
      setState(807);
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

//----------------- Xor_exprContext ------------------------------------------------------------------

pyxasmParser::Xor_exprContext::Xor_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::And_exprContext *> pyxasmParser::Xor_exprContext::and_expr() {
  return getRuleContexts<pyxasmParser::And_exprContext>();
}

pyxasmParser::And_exprContext* pyxasmParser::Xor_exprContext::and_expr(size_t i) {
  return getRuleContext<pyxasmParser::And_exprContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Xor_exprContext::XOR() {
  return getTokens(pyxasmParser::XOR);
}

tree::TerminalNode* pyxasmParser::Xor_exprContext::XOR(size_t i) {
  return getToken(pyxasmParser::XOR, i);
}


size_t pyxasmParser::Xor_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleXor_expr;
}

void pyxasmParser::Xor_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterXor_expr(this);
}

void pyxasmParser::Xor_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitXor_expr(this);
}


antlrcpp::Any pyxasmParser::Xor_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitXor_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Xor_exprContext* pyxasmParser::xor_expr() {
  Xor_exprContext *_localctx = _tracker.createInstance<Xor_exprContext>(_ctx, getState());
  enterRule(_localctx, 120, pyxasmParser::RuleXor_expr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(808);
    and_expr();
    setState(813);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::XOR) {
      setState(809);
      match(pyxasmParser::XOR);
      setState(810);
      and_expr();
      setState(815);
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

//----------------- And_exprContext ------------------------------------------------------------------

pyxasmParser::And_exprContext::And_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Shift_exprContext *> pyxasmParser::And_exprContext::shift_expr() {
  return getRuleContexts<pyxasmParser::Shift_exprContext>();
}

pyxasmParser::Shift_exprContext* pyxasmParser::And_exprContext::shift_expr(size_t i) {
  return getRuleContext<pyxasmParser::Shift_exprContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::And_exprContext::AND_OP() {
  return getTokens(pyxasmParser::AND_OP);
}

tree::TerminalNode* pyxasmParser::And_exprContext::AND_OP(size_t i) {
  return getToken(pyxasmParser::AND_OP, i);
}


size_t pyxasmParser::And_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleAnd_expr;
}

void pyxasmParser::And_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnd_expr(this);
}

void pyxasmParser::And_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnd_expr(this);
}


antlrcpp::Any pyxasmParser::And_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAnd_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::And_exprContext* pyxasmParser::and_expr() {
  And_exprContext *_localctx = _tracker.createInstance<And_exprContext>(_ctx, getState());
  enterRule(_localctx, 122, pyxasmParser::RuleAnd_expr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(816);
    shift_expr();
    setState(821);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::AND_OP) {
      setState(817);
      match(pyxasmParser::AND_OP);
      setState(818);
      shift_expr();
      setState(823);
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

//----------------- Shift_exprContext ------------------------------------------------------------------

pyxasmParser::Shift_exprContext::Shift_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::Arith_exprContext *> pyxasmParser::Shift_exprContext::arith_expr() {
  return getRuleContexts<pyxasmParser::Arith_exprContext>();
}

pyxasmParser::Arith_exprContext* pyxasmParser::Shift_exprContext::arith_expr(size_t i) {
  return getRuleContext<pyxasmParser::Arith_exprContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Shift_exprContext::LEFT_SHIFT() {
  return getTokens(pyxasmParser::LEFT_SHIFT);
}

tree::TerminalNode* pyxasmParser::Shift_exprContext::LEFT_SHIFT(size_t i) {
  return getToken(pyxasmParser::LEFT_SHIFT, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Shift_exprContext::RIGHT_SHIFT() {
  return getTokens(pyxasmParser::RIGHT_SHIFT);
}

tree::TerminalNode* pyxasmParser::Shift_exprContext::RIGHT_SHIFT(size_t i) {
  return getToken(pyxasmParser::RIGHT_SHIFT, i);
}


size_t pyxasmParser::Shift_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleShift_expr;
}

void pyxasmParser::Shift_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShift_expr(this);
}

void pyxasmParser::Shift_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShift_expr(this);
}


antlrcpp::Any pyxasmParser::Shift_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitShift_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Shift_exprContext* pyxasmParser::shift_expr() {
  Shift_exprContext *_localctx = _tracker.createInstance<Shift_exprContext>(_ctx, getState());
  enterRule(_localctx, 124, pyxasmParser::RuleShift_expr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(824);
    arith_expr();
    setState(829);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::LEFT_SHIFT

    || _la == pyxasmParser::RIGHT_SHIFT) {
      setState(825);
      _la = _input->LA(1);
      if (!(_la == pyxasmParser::LEFT_SHIFT

      || _la == pyxasmParser::RIGHT_SHIFT)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(826);
      arith_expr();
      setState(831);
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

//----------------- Arith_exprContext ------------------------------------------------------------------

pyxasmParser::Arith_exprContext::Arith_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TermContext *> pyxasmParser::Arith_exprContext::term() {
  return getRuleContexts<pyxasmParser::TermContext>();
}

pyxasmParser::TermContext* pyxasmParser::Arith_exprContext::term(size_t i) {
  return getRuleContext<pyxasmParser::TermContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Arith_exprContext::ADD() {
  return getTokens(pyxasmParser::ADD);
}

tree::TerminalNode* pyxasmParser::Arith_exprContext::ADD(size_t i) {
  return getToken(pyxasmParser::ADD, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::Arith_exprContext::MINUS() {
  return getTokens(pyxasmParser::MINUS);
}

tree::TerminalNode* pyxasmParser::Arith_exprContext::MINUS(size_t i) {
  return getToken(pyxasmParser::MINUS, i);
}


size_t pyxasmParser::Arith_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleArith_expr;
}

void pyxasmParser::Arith_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArith_expr(this);
}

void pyxasmParser::Arith_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArith_expr(this);
}


antlrcpp::Any pyxasmParser::Arith_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitArith_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Arith_exprContext* pyxasmParser::arith_expr() {
  Arith_exprContext *_localctx = _tracker.createInstance<Arith_exprContext>(_ctx, getState());
  enterRule(_localctx, 126, pyxasmParser::RuleArith_expr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(832);
    term();
    setState(837);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == pyxasmParser::ADD

    || _la == pyxasmParser::MINUS) {
      setState(833);
      _la = _input->LA(1);
      if (!(_la == pyxasmParser::ADD

      || _la == pyxasmParser::MINUS)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(834);
      term();
      setState(839);
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

//----------------- TermContext ------------------------------------------------------------------

pyxasmParser::TermContext::TermContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::FactorContext *> pyxasmParser::TermContext::factor() {
  return getRuleContexts<pyxasmParser::FactorContext>();
}

pyxasmParser::FactorContext* pyxasmParser::TermContext::factor(size_t i) {
  return getRuleContext<pyxasmParser::FactorContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::TermContext::STAR() {
  return getTokens(pyxasmParser::STAR);
}

tree::TerminalNode* pyxasmParser::TermContext::STAR(size_t i) {
  return getToken(pyxasmParser::STAR, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::TermContext::AT() {
  return getTokens(pyxasmParser::AT);
}

tree::TerminalNode* pyxasmParser::TermContext::AT(size_t i) {
  return getToken(pyxasmParser::AT, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::TermContext::DIV() {
  return getTokens(pyxasmParser::DIV);
}

tree::TerminalNode* pyxasmParser::TermContext::DIV(size_t i) {
  return getToken(pyxasmParser::DIV, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::TermContext::MOD() {
  return getTokens(pyxasmParser::MOD);
}

tree::TerminalNode* pyxasmParser::TermContext::MOD(size_t i) {
  return getToken(pyxasmParser::MOD, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::TermContext::IDIV() {
  return getTokens(pyxasmParser::IDIV);
}

tree::TerminalNode* pyxasmParser::TermContext::IDIV(size_t i) {
  return getToken(pyxasmParser::IDIV, i);
}


size_t pyxasmParser::TermContext::getRuleIndex() const {
  return pyxasmParser::RuleTerm;
}

void pyxasmParser::TermContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTerm(this);
}

void pyxasmParser::TermContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTerm(this);
}


antlrcpp::Any pyxasmParser::TermContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTerm(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::TermContext* pyxasmParser::term() {
  TermContext *_localctx = _tracker.createInstance<TermContext>(_ctx, getState());
  enterRule(_localctx, 128, pyxasmParser::RuleTerm);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(840);
    factor();
    setState(845);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (((((_la - 51) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 51)) & ((1ULL << (pyxasmParser::STAR - 51))
      | (1ULL << (pyxasmParser::DIV - 51))
      | (1ULL << (pyxasmParser::MOD - 51))
      | (1ULL << (pyxasmParser::IDIV - 51))
      | (1ULL << (pyxasmParser::AT - 51)))) != 0)) {
      setState(841);
      _la = _input->LA(1);
      if (!(((((_la - 51) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 51)) & ((1ULL << (pyxasmParser::STAR - 51))
        | (1ULL << (pyxasmParser::DIV - 51))
        | (1ULL << (pyxasmParser::MOD - 51))
        | (1ULL << (pyxasmParser::IDIV - 51))
        | (1ULL << (pyxasmParser::AT - 51)))) != 0))) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(842);
      factor();
      setState(847);
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

//----------------- FactorContext ------------------------------------------------------------------

pyxasmParser::FactorContext::FactorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::FactorContext* pyxasmParser::FactorContext::factor() {
  return getRuleContext<pyxasmParser::FactorContext>(0);
}

tree::TerminalNode* pyxasmParser::FactorContext::ADD() {
  return getToken(pyxasmParser::ADD, 0);
}

tree::TerminalNode* pyxasmParser::FactorContext::MINUS() {
  return getToken(pyxasmParser::MINUS, 0);
}

tree::TerminalNode* pyxasmParser::FactorContext::NOT_OP() {
  return getToken(pyxasmParser::NOT_OP, 0);
}

pyxasmParser::PowerContext* pyxasmParser::FactorContext::power() {
  return getRuleContext<pyxasmParser::PowerContext>(0);
}


size_t pyxasmParser::FactorContext::getRuleIndex() const {
  return pyxasmParser::RuleFactor;
}

void pyxasmParser::FactorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFactor(this);
}

void pyxasmParser::FactorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFactor(this);
}


antlrcpp::Any pyxasmParser::FactorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitFactor(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::FactorContext* pyxasmParser::factor() {
  FactorContext *_localctx = _tracker.createInstance<FactorContext>(_ctx, getState());
  enterRule(_localctx, 130, pyxasmParser::RuleFactor);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(851);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP: {
        enterOuterAlt(_localctx, 1);
        setState(848);
        _la = _input->LA(1);
        if (!(((((_la - 66) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
          | (1ULL << (pyxasmParser::MINUS - 66))
          | (1ULL << (pyxasmParser::NOT_OP - 66)))) != 0))) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(849);
        factor();
        break;
      }

      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 2);
        setState(850);
        power();
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

//----------------- PowerContext ------------------------------------------------------------------

pyxasmParser::PowerContext::PowerContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Atom_exprContext* pyxasmParser::PowerContext::atom_expr() {
  return getRuleContext<pyxasmParser::Atom_exprContext>(0);
}

tree::TerminalNode* pyxasmParser::PowerContext::POWER() {
  return getToken(pyxasmParser::POWER, 0);
}

pyxasmParser::FactorContext* pyxasmParser::PowerContext::factor() {
  return getRuleContext<pyxasmParser::FactorContext>(0);
}


size_t pyxasmParser::PowerContext::getRuleIndex() const {
  return pyxasmParser::RulePower;
}

void pyxasmParser::PowerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPower(this);
}

void pyxasmParser::PowerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPower(this);
}


antlrcpp::Any pyxasmParser::PowerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitPower(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::PowerContext* pyxasmParser::power() {
  PowerContext *_localctx = _tracker.createInstance<PowerContext>(_ctx, getState());
  enterRule(_localctx, 132, pyxasmParser::RulePower);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(853);
    atom_expr();
    setState(856);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::POWER) {
      setState(854);
      match(pyxasmParser::POWER);
      setState(855);
      factor();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Atom_exprContext ------------------------------------------------------------------

pyxasmParser::Atom_exprContext::Atom_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::AtomContext* pyxasmParser::Atom_exprContext::atom() {
  return getRuleContext<pyxasmParser::AtomContext>(0);
}

tree::TerminalNode* pyxasmParser::Atom_exprContext::AWAIT() {
  return getToken(pyxasmParser::AWAIT, 0);
}

std::vector<pyxasmParser::TrailerContext *> pyxasmParser::Atom_exprContext::trailer() {
  return getRuleContexts<pyxasmParser::TrailerContext>();
}

pyxasmParser::TrailerContext* pyxasmParser::Atom_exprContext::trailer(size_t i) {
  return getRuleContext<pyxasmParser::TrailerContext>(i);
}


size_t pyxasmParser::Atom_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleAtom_expr;
}

void pyxasmParser::Atom_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAtom_expr(this);
}

void pyxasmParser::Atom_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAtom_expr(this);
}


antlrcpp::Any pyxasmParser::Atom_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAtom_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Atom_exprContext* pyxasmParser::atom_expr() {
  Atom_exprContext *_localctx = _tracker.createInstance<Atom_exprContext>(_ctx, getState());
  enterRule(_localctx, 134, pyxasmParser::RuleAtom_expr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(859);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::AWAIT) {
      setState(858);
      match(pyxasmParser::AWAIT);
    }
    setState(861);
    atom();
    setState(865);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::DOT)
      | (1ULL << pyxasmParser::OPEN_PAREN)
      | (1ULL << pyxasmParser::OPEN_BRACK))) != 0)) {
      setState(862);
      trailer();
      setState(867);
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

//----------------- AtomContext ------------------------------------------------------------------

pyxasmParser::AtomContext::AtomContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::AtomContext::OPEN_PAREN() {
  return getToken(pyxasmParser::OPEN_PAREN, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::CLOSE_PAREN() {
  return getToken(pyxasmParser::CLOSE_PAREN, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::OPEN_BRACK() {
  return getToken(pyxasmParser::OPEN_BRACK, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::CLOSE_BRACK() {
  return getToken(pyxasmParser::CLOSE_BRACK, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::OPEN_BRACE() {
  return getToken(pyxasmParser::OPEN_BRACE, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::CLOSE_BRACE() {
  return getToken(pyxasmParser::CLOSE_BRACE, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::NUMBER() {
  return getToken(pyxasmParser::NUMBER, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::ELLIPSIS() {
  return getToken(pyxasmParser::ELLIPSIS, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::NONE() {
  return getToken(pyxasmParser::NONE, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::TRUE() {
  return getToken(pyxasmParser::TRUE, 0);
}

tree::TerminalNode* pyxasmParser::AtomContext::FALSE() {
  return getToken(pyxasmParser::FALSE, 0);
}

pyxasmParser::Yield_exprContext* pyxasmParser::AtomContext::yield_expr() {
  return getRuleContext<pyxasmParser::Yield_exprContext>(0);
}

pyxasmParser::Testlist_compContext* pyxasmParser::AtomContext::testlist_comp() {
  return getRuleContext<pyxasmParser::Testlist_compContext>(0);
}

pyxasmParser::DictorsetmakerContext* pyxasmParser::AtomContext::dictorsetmaker() {
  return getRuleContext<pyxasmParser::DictorsetmakerContext>(0);
}

std::vector<tree::TerminalNode *> pyxasmParser::AtomContext::STRING() {
  return getTokens(pyxasmParser::STRING);
}

tree::TerminalNode* pyxasmParser::AtomContext::STRING(size_t i) {
  return getToken(pyxasmParser::STRING, i);
}


size_t pyxasmParser::AtomContext::getRuleIndex() const {
  return pyxasmParser::RuleAtom;
}

void pyxasmParser::AtomContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAtom(this);
}

void pyxasmParser::AtomContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAtom(this);
}


antlrcpp::Any pyxasmParser::AtomContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitAtom(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::AtomContext* pyxasmParser::atom() {
  AtomContext *_localctx = _tracker.createInstance<AtomContext>(_ctx, getState());
  enterRule(_localctx, 136, pyxasmParser::RuleAtom);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(895);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::OPEN_PAREN: {
        setState(868);
        match(pyxasmParser::OPEN_PAREN);
        setState(871);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case pyxasmParser::YIELD: {
            setState(869);
            yield_expr();
            break;
          }

          case pyxasmParser::STRING:
          case pyxasmParser::NUMBER:
          case pyxasmParser::LAMBDA:
          case pyxasmParser::NOT:
          case pyxasmParser::NONE:
          case pyxasmParser::TRUE:
          case pyxasmParser::FALSE:
          case pyxasmParser::AWAIT:
          case pyxasmParser::NAME:
          case pyxasmParser::ELLIPSIS:
          case pyxasmParser::STAR:
          case pyxasmParser::OPEN_PAREN:
          case pyxasmParser::OPEN_BRACK:
          case pyxasmParser::ADD:
          case pyxasmParser::MINUS:
          case pyxasmParser::NOT_OP:
          case pyxasmParser::OPEN_BRACE: {
            setState(870);
            testlist_comp();
            break;
          }

          case pyxasmParser::CLOSE_PAREN: {
            break;
          }

        default:
          break;
        }
        setState(873);
        match(pyxasmParser::CLOSE_PAREN);
        break;
      }

      case pyxasmParser::OPEN_BRACK: {
        setState(874);
        match(pyxasmParser::OPEN_BRACK);
        setState(876);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
          | (1ULL << pyxasmParser::NUMBER)
          | (1ULL << pyxasmParser::LAMBDA)
          | (1ULL << pyxasmParser::NOT)
          | (1ULL << pyxasmParser::NONE)
          | (1ULL << pyxasmParser::TRUE)
          | (1ULL << pyxasmParser::FALSE)
          | (1ULL << pyxasmParser::AWAIT)
          | (1ULL << pyxasmParser::NAME)
          | (1ULL << pyxasmParser::ELLIPSIS)
          | (1ULL << pyxasmParser::STAR)
          | (1ULL << pyxasmParser::OPEN_PAREN)
          | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
          | (1ULL << (pyxasmParser::MINUS - 66))
          | (1ULL << (pyxasmParser::NOT_OP - 66))
          | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
          setState(875);
          testlist_comp();
        }
        setState(878);
        match(pyxasmParser::CLOSE_BRACK);
        break;
      }

      case pyxasmParser::OPEN_BRACE: {
        setState(879);
        match(pyxasmParser::OPEN_BRACE);
        setState(881);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
          | (1ULL << pyxasmParser::NUMBER)
          | (1ULL << pyxasmParser::LAMBDA)
          | (1ULL << pyxasmParser::NOT)
          | (1ULL << pyxasmParser::NONE)
          | (1ULL << pyxasmParser::TRUE)
          | (1ULL << pyxasmParser::FALSE)
          | (1ULL << pyxasmParser::AWAIT)
          | (1ULL << pyxasmParser::NAME)
          | (1ULL << pyxasmParser::ELLIPSIS)
          | (1ULL << pyxasmParser::STAR)
          | (1ULL << pyxasmParser::OPEN_PAREN)
          | (1ULL << pyxasmParser::POWER)
          | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
          | (1ULL << (pyxasmParser::MINUS - 66))
          | (1ULL << (pyxasmParser::NOT_OP - 66))
          | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
          setState(880);
          dictorsetmaker();
        }
        setState(883);
        match(pyxasmParser::CLOSE_BRACE);
        break;
      }

      case pyxasmParser::NAME: {
        setState(884);
        match(pyxasmParser::NAME);
        break;
      }

      case pyxasmParser::NUMBER: {
        setState(885);
        match(pyxasmParser::NUMBER);
        break;
      }

      case pyxasmParser::STRING: {
        setState(887); 
        _errHandler->sync(this);
        _la = _input->LA(1);
        do {
          setState(886);
          match(pyxasmParser::STRING);
          setState(889); 
          _errHandler->sync(this);
          _la = _input->LA(1);
        } while (_la == pyxasmParser::STRING);
        break;
      }

      case pyxasmParser::ELLIPSIS: {
        setState(891);
        match(pyxasmParser::ELLIPSIS);
        break;
      }

      case pyxasmParser::NONE: {
        setState(892);
        match(pyxasmParser::NONE);
        break;
      }

      case pyxasmParser::TRUE: {
        setState(893);
        match(pyxasmParser::TRUE);
        break;
      }

      case pyxasmParser::FALSE: {
        setState(894);
        match(pyxasmParser::FALSE);
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

//----------------- Testlist_compContext ------------------------------------------------------------------

pyxasmParser::Testlist_compContext::Testlist_compContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::Testlist_compContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::Testlist_compContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

std::vector<pyxasmParser::Star_exprContext *> pyxasmParser::Testlist_compContext::star_expr() {
  return getRuleContexts<pyxasmParser::Star_exprContext>();
}

pyxasmParser::Star_exprContext* pyxasmParser::Testlist_compContext::star_expr(size_t i) {
  return getRuleContext<pyxasmParser::Star_exprContext>(i);
}

pyxasmParser::Comp_forContext* pyxasmParser::Testlist_compContext::comp_for() {
  return getRuleContext<pyxasmParser::Comp_forContext>(0);
}

std::vector<tree::TerminalNode *> pyxasmParser::Testlist_compContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::Testlist_compContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::Testlist_compContext::getRuleIndex() const {
  return pyxasmParser::RuleTestlist_comp;
}

void pyxasmParser::Testlist_compContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTestlist_comp(this);
}

void pyxasmParser::Testlist_compContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTestlist_comp(this);
}


antlrcpp::Any pyxasmParser::Testlist_compContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTestlist_comp(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Testlist_compContext* pyxasmParser::testlist_comp() {
  Testlist_compContext *_localctx = _tracker.createInstance<Testlist_compContext>(_ctx, getState());
  enterRule(_localctx, 138, pyxasmParser::RuleTestlist_comp);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(899);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::LAMBDA:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        setState(897);
        test();
        break;
      }

      case pyxasmParser::STAR: {
        setState(898);
        star_expr();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(915);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::FOR:
      case pyxasmParser::ASYNC: {
        setState(901);
        comp_for();
        break;
      }

      case pyxasmParser::CLOSE_PAREN:
      case pyxasmParser::COMMA:
      case pyxasmParser::CLOSE_BRACK: {
        setState(909);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 125, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(902);
            match(pyxasmParser::COMMA);
            setState(905);
            _errHandler->sync(this);
            switch (_input->LA(1)) {
              case pyxasmParser::STRING:
              case pyxasmParser::NUMBER:
              case pyxasmParser::LAMBDA:
              case pyxasmParser::NOT:
              case pyxasmParser::NONE:
              case pyxasmParser::TRUE:
              case pyxasmParser::FALSE:
              case pyxasmParser::AWAIT:
              case pyxasmParser::NAME:
              case pyxasmParser::ELLIPSIS:
              case pyxasmParser::OPEN_PAREN:
              case pyxasmParser::OPEN_BRACK:
              case pyxasmParser::ADD:
              case pyxasmParser::MINUS:
              case pyxasmParser::NOT_OP:
              case pyxasmParser::OPEN_BRACE: {
                setState(903);
                test();
                break;
              }

              case pyxasmParser::STAR: {
                setState(904);
                star_expr();
                break;
              }

            default:
              throw NoViableAltException(this);
            } 
          }
          setState(911);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 125, _ctx);
        }
        setState(913);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == pyxasmParser::COMMA) {
          setState(912);
          match(pyxasmParser::COMMA);
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

//----------------- TrailerContext ------------------------------------------------------------------

pyxasmParser::TrailerContext::TrailerContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::TrailerContext::OPEN_PAREN() {
  return getToken(pyxasmParser::OPEN_PAREN, 0);
}

tree::TerminalNode* pyxasmParser::TrailerContext::CLOSE_PAREN() {
  return getToken(pyxasmParser::CLOSE_PAREN, 0);
}

pyxasmParser::ArglistContext* pyxasmParser::TrailerContext::arglist() {
  return getRuleContext<pyxasmParser::ArglistContext>(0);
}

tree::TerminalNode* pyxasmParser::TrailerContext::OPEN_BRACK() {
  return getToken(pyxasmParser::OPEN_BRACK, 0);
}

pyxasmParser::SubscriptlistContext* pyxasmParser::TrailerContext::subscriptlist() {
  return getRuleContext<pyxasmParser::SubscriptlistContext>(0);
}

tree::TerminalNode* pyxasmParser::TrailerContext::CLOSE_BRACK() {
  return getToken(pyxasmParser::CLOSE_BRACK, 0);
}

tree::TerminalNode* pyxasmParser::TrailerContext::DOT() {
  return getToken(pyxasmParser::DOT, 0);
}

tree::TerminalNode* pyxasmParser::TrailerContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}


size_t pyxasmParser::TrailerContext::getRuleIndex() const {
  return pyxasmParser::RuleTrailer;
}

void pyxasmParser::TrailerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTrailer(this);
}

void pyxasmParser::TrailerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTrailer(this);
}


antlrcpp::Any pyxasmParser::TrailerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTrailer(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::TrailerContext* pyxasmParser::trailer() {
  TrailerContext *_localctx = _tracker.createInstance<TrailerContext>(_ctx, getState());
  enterRule(_localctx, 140, pyxasmParser::RuleTrailer);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(928);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::OPEN_PAREN: {
        enterOuterAlt(_localctx, 1);
        setState(917);
        match(pyxasmParser::OPEN_PAREN);
        setState(919);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
          | (1ULL << pyxasmParser::NUMBER)
          | (1ULL << pyxasmParser::LAMBDA)
          | (1ULL << pyxasmParser::NOT)
          | (1ULL << pyxasmParser::NONE)
          | (1ULL << pyxasmParser::TRUE)
          | (1ULL << pyxasmParser::FALSE)
          | (1ULL << pyxasmParser::AWAIT)
          | (1ULL << pyxasmParser::NAME)
          | (1ULL << pyxasmParser::ELLIPSIS)
          | (1ULL << pyxasmParser::STAR)
          | (1ULL << pyxasmParser::OPEN_PAREN)
          | (1ULL << pyxasmParser::POWER)
          | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
          | (1ULL << (pyxasmParser::MINUS - 66))
          | (1ULL << (pyxasmParser::NOT_OP - 66))
          | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
          setState(918);
          arglist();
        }
        setState(921);
        match(pyxasmParser::CLOSE_PAREN);
        break;
      }

      case pyxasmParser::OPEN_BRACK: {
        enterOuterAlt(_localctx, 2);
        setState(922);
        match(pyxasmParser::OPEN_BRACK);
        setState(923);
        subscriptlist();
        setState(924);
        match(pyxasmParser::CLOSE_BRACK);
        break;
      }

      case pyxasmParser::DOT: {
        enterOuterAlt(_localctx, 3);
        setState(926);
        match(pyxasmParser::DOT);
        setState(927);
        match(pyxasmParser::NAME);
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

//----------------- SubscriptlistContext ------------------------------------------------------------------

pyxasmParser::SubscriptlistContext::SubscriptlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::SubscriptContext *> pyxasmParser::SubscriptlistContext::subscript() {
  return getRuleContexts<pyxasmParser::SubscriptContext>();
}

pyxasmParser::SubscriptContext* pyxasmParser::SubscriptlistContext::subscript(size_t i) {
  return getRuleContext<pyxasmParser::SubscriptContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::SubscriptlistContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::SubscriptlistContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::SubscriptlistContext::getRuleIndex() const {
  return pyxasmParser::RuleSubscriptlist;
}

void pyxasmParser::SubscriptlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubscriptlist(this);
}

void pyxasmParser::SubscriptlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubscriptlist(this);
}


antlrcpp::Any pyxasmParser::SubscriptlistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitSubscriptlist(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::SubscriptlistContext* pyxasmParser::subscriptlist() {
  SubscriptlistContext *_localctx = _tracker.createInstance<SubscriptlistContext>(_ctx, getState());
  enterRule(_localctx, 142, pyxasmParser::RuleSubscriptlist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(930);
    subscript();
    setState(935);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 130, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(931);
        match(pyxasmParser::COMMA);
        setState(932);
        subscript(); 
      }
      setState(937);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 130, _ctx);
    }
    setState(939);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COMMA) {
      setState(938);
      match(pyxasmParser::COMMA);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SubscriptContext ------------------------------------------------------------------

pyxasmParser::SubscriptContext::SubscriptContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::SubscriptContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::SubscriptContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

tree::TerminalNode* pyxasmParser::SubscriptContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::SliceopContext* pyxasmParser::SubscriptContext::sliceop() {
  return getRuleContext<pyxasmParser::SliceopContext>(0);
}


size_t pyxasmParser::SubscriptContext::getRuleIndex() const {
  return pyxasmParser::RuleSubscript;
}

void pyxasmParser::SubscriptContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubscript(this);
}

void pyxasmParser::SubscriptContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubscript(this);
}


antlrcpp::Any pyxasmParser::SubscriptContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitSubscript(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::SubscriptContext* pyxasmParser::subscript() {
  SubscriptContext *_localctx = _tracker.createInstance<SubscriptContext>(_ctx, getState());
  enterRule(_localctx, 144, pyxasmParser::RuleSubscript);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(952);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 135, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(941);
      test();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(943);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
        | (1ULL << pyxasmParser::NUMBER)
        | (1ULL << pyxasmParser::LAMBDA)
        | (1ULL << pyxasmParser::NOT)
        | (1ULL << pyxasmParser::NONE)
        | (1ULL << pyxasmParser::TRUE)
        | (1ULL << pyxasmParser::FALSE)
        | (1ULL << pyxasmParser::AWAIT)
        | (1ULL << pyxasmParser::NAME)
        | (1ULL << pyxasmParser::ELLIPSIS)
        | (1ULL << pyxasmParser::OPEN_PAREN)
        | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
        | (1ULL << (pyxasmParser::MINUS - 66))
        | (1ULL << (pyxasmParser::NOT_OP - 66))
        | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
        setState(942);
        test();
      }
      setState(945);
      match(pyxasmParser::COLON);
      setState(947);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
        | (1ULL << pyxasmParser::NUMBER)
        | (1ULL << pyxasmParser::LAMBDA)
        | (1ULL << pyxasmParser::NOT)
        | (1ULL << pyxasmParser::NONE)
        | (1ULL << pyxasmParser::TRUE)
        | (1ULL << pyxasmParser::FALSE)
        | (1ULL << pyxasmParser::AWAIT)
        | (1ULL << pyxasmParser::NAME)
        | (1ULL << pyxasmParser::ELLIPSIS)
        | (1ULL << pyxasmParser::OPEN_PAREN)
        | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
        | (1ULL << (pyxasmParser::MINUS - 66))
        | (1ULL << (pyxasmParser::NOT_OP - 66))
        | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
        setState(946);
        test();
      }
      setState(950);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == pyxasmParser::COLON) {
        setState(949);
        sliceop();
      }
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

//----------------- SliceopContext ------------------------------------------------------------------

pyxasmParser::SliceopContext::SliceopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::SliceopContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::TestContext* pyxasmParser::SliceopContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}


size_t pyxasmParser::SliceopContext::getRuleIndex() const {
  return pyxasmParser::RuleSliceop;
}

void pyxasmParser::SliceopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSliceop(this);
}

void pyxasmParser::SliceopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSliceop(this);
}


antlrcpp::Any pyxasmParser::SliceopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitSliceop(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::SliceopContext* pyxasmParser::sliceop() {
  SliceopContext *_localctx = _tracker.createInstance<SliceopContext>(_ctx, getState());
  enterRule(_localctx, 146, pyxasmParser::RuleSliceop);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(954);
    match(pyxasmParser::COLON);
    setState(956);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
      | (1ULL << pyxasmParser::NUMBER)
      | (1ULL << pyxasmParser::LAMBDA)
      | (1ULL << pyxasmParser::NOT)
      | (1ULL << pyxasmParser::NONE)
      | (1ULL << pyxasmParser::TRUE)
      | (1ULL << pyxasmParser::FALSE)
      | (1ULL << pyxasmParser::AWAIT)
      | (1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::ELLIPSIS)
      | (1ULL << pyxasmParser::OPEN_PAREN)
      | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
      | (1ULL << (pyxasmParser::MINUS - 66))
      | (1ULL << (pyxasmParser::NOT_OP - 66))
      | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
      setState(955);
      test();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprlistContext ------------------------------------------------------------------

pyxasmParser::ExprlistContext::ExprlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::ExprContext *> pyxasmParser::ExprlistContext::expr() {
  return getRuleContexts<pyxasmParser::ExprContext>();
}

pyxasmParser::ExprContext* pyxasmParser::ExprlistContext::expr(size_t i) {
  return getRuleContext<pyxasmParser::ExprContext>(i);
}

std::vector<pyxasmParser::Star_exprContext *> pyxasmParser::ExprlistContext::star_expr() {
  return getRuleContexts<pyxasmParser::Star_exprContext>();
}

pyxasmParser::Star_exprContext* pyxasmParser::ExprlistContext::star_expr(size_t i) {
  return getRuleContext<pyxasmParser::Star_exprContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::ExprlistContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::ExprlistContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::ExprlistContext::getRuleIndex() const {
  return pyxasmParser::RuleExprlist;
}

void pyxasmParser::ExprlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExprlist(this);
}

void pyxasmParser::ExprlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExprlist(this);
}


antlrcpp::Any pyxasmParser::ExprlistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitExprlist(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::ExprlistContext* pyxasmParser::exprlist() {
  ExprlistContext *_localctx = _tracker.createInstance<ExprlistContext>(_ctx, getState());
  enterRule(_localctx, 148, pyxasmParser::RuleExprlist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(960);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        setState(958);
        expr();
        break;
      }

      case pyxasmParser::STAR: {
        setState(959);
        star_expr();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(969);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 139, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(962);
        match(pyxasmParser::COMMA);
        setState(965);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case pyxasmParser::STRING:
          case pyxasmParser::NUMBER:
          case pyxasmParser::NONE:
          case pyxasmParser::TRUE:
          case pyxasmParser::FALSE:
          case pyxasmParser::AWAIT:
          case pyxasmParser::NAME:
          case pyxasmParser::ELLIPSIS:
          case pyxasmParser::OPEN_PAREN:
          case pyxasmParser::OPEN_BRACK:
          case pyxasmParser::ADD:
          case pyxasmParser::MINUS:
          case pyxasmParser::NOT_OP:
          case pyxasmParser::OPEN_BRACE: {
            setState(963);
            expr();
            break;
          }

          case pyxasmParser::STAR: {
            setState(964);
            star_expr();
            break;
          }

        default:
          throw NoViableAltException(this);
        } 
      }
      setState(971);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 139, _ctx);
    }
    setState(973);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COMMA) {
      setState(972);
      match(pyxasmParser::COMMA);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TestlistContext ------------------------------------------------------------------

pyxasmParser::TestlistContext::TestlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::TestlistContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::TestlistContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::TestlistContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::TestlistContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::TestlistContext::getRuleIndex() const {
  return pyxasmParser::RuleTestlist;
}

void pyxasmParser::TestlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTestlist(this);
}

void pyxasmParser::TestlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTestlist(this);
}


antlrcpp::Any pyxasmParser::TestlistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitTestlist(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::TestlistContext* pyxasmParser::testlist() {
  TestlistContext *_localctx = _tracker.createInstance<TestlistContext>(_ctx, getState());
  enterRule(_localctx, 150, pyxasmParser::RuleTestlist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(975);
    test();
    setState(980);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 141, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(976);
        match(pyxasmParser::COMMA);
        setState(977);
        test(); 
      }
      setState(982);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 141, _ctx);
    }
    setState(984);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COMMA) {
      setState(983);
      match(pyxasmParser::COMMA);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DictorsetmakerContext ------------------------------------------------------------------

pyxasmParser::DictorsetmakerContext::DictorsetmakerContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::DictorsetmakerContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::DictorsetmakerContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::DictorsetmakerContext::COLON() {
  return getTokens(pyxasmParser::COLON);
}

tree::TerminalNode* pyxasmParser::DictorsetmakerContext::COLON(size_t i) {
  return getToken(pyxasmParser::COLON, i);
}

std::vector<tree::TerminalNode *> pyxasmParser::DictorsetmakerContext::POWER() {
  return getTokens(pyxasmParser::POWER);
}

tree::TerminalNode* pyxasmParser::DictorsetmakerContext::POWER(size_t i) {
  return getToken(pyxasmParser::POWER, i);
}

std::vector<pyxasmParser::ExprContext *> pyxasmParser::DictorsetmakerContext::expr() {
  return getRuleContexts<pyxasmParser::ExprContext>();
}

pyxasmParser::ExprContext* pyxasmParser::DictorsetmakerContext::expr(size_t i) {
  return getRuleContext<pyxasmParser::ExprContext>(i);
}

pyxasmParser::Comp_forContext* pyxasmParser::DictorsetmakerContext::comp_for() {
  return getRuleContext<pyxasmParser::Comp_forContext>(0);
}

std::vector<pyxasmParser::Star_exprContext *> pyxasmParser::DictorsetmakerContext::star_expr() {
  return getRuleContexts<pyxasmParser::Star_exprContext>();
}

pyxasmParser::Star_exprContext* pyxasmParser::DictorsetmakerContext::star_expr(size_t i) {
  return getRuleContext<pyxasmParser::Star_exprContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::DictorsetmakerContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::DictorsetmakerContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::DictorsetmakerContext::getRuleIndex() const {
  return pyxasmParser::RuleDictorsetmaker;
}

void pyxasmParser::DictorsetmakerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDictorsetmaker(this);
}

void pyxasmParser::DictorsetmakerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDictorsetmaker(this);
}


antlrcpp::Any pyxasmParser::DictorsetmakerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitDictorsetmaker(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::DictorsetmakerContext* pyxasmParser::dictorsetmaker() {
  DictorsetmakerContext *_localctx = _tracker.createInstance<DictorsetmakerContext>(_ctx, getState());
  enterRule(_localctx, 152, pyxasmParser::RuleDictorsetmaker);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1034);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 153, _ctx)) {
    case 1: {
      setState(992);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case pyxasmParser::STRING:
        case pyxasmParser::NUMBER:
        case pyxasmParser::LAMBDA:
        case pyxasmParser::NOT:
        case pyxasmParser::NONE:
        case pyxasmParser::TRUE:
        case pyxasmParser::FALSE:
        case pyxasmParser::AWAIT:
        case pyxasmParser::NAME:
        case pyxasmParser::ELLIPSIS:
        case pyxasmParser::OPEN_PAREN:
        case pyxasmParser::OPEN_BRACK:
        case pyxasmParser::ADD:
        case pyxasmParser::MINUS:
        case pyxasmParser::NOT_OP:
        case pyxasmParser::OPEN_BRACE: {
          setState(986);
          test();
          setState(987);
          match(pyxasmParser::COLON);
          setState(988);
          test();
          break;
        }

        case pyxasmParser::POWER: {
          setState(990);
          match(pyxasmParser::POWER);
          setState(991);
          expr();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(1012);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case pyxasmParser::FOR:
        case pyxasmParser::ASYNC: {
          setState(994);
          comp_for();
          break;
        }

        case pyxasmParser::COMMA:
        case pyxasmParser::CLOSE_BRACE: {
          setState(1006);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 145, _ctx);
          while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
            if (alt == 1) {
              setState(995);
              match(pyxasmParser::COMMA);
              setState(1002);
              _errHandler->sync(this);
              switch (_input->LA(1)) {
                case pyxasmParser::STRING:
                case pyxasmParser::NUMBER:
                case pyxasmParser::LAMBDA:
                case pyxasmParser::NOT:
                case pyxasmParser::NONE:
                case pyxasmParser::TRUE:
                case pyxasmParser::FALSE:
                case pyxasmParser::AWAIT:
                case pyxasmParser::NAME:
                case pyxasmParser::ELLIPSIS:
                case pyxasmParser::OPEN_PAREN:
                case pyxasmParser::OPEN_BRACK:
                case pyxasmParser::ADD:
                case pyxasmParser::MINUS:
                case pyxasmParser::NOT_OP:
                case pyxasmParser::OPEN_BRACE: {
                  setState(996);
                  test();
                  setState(997);
                  match(pyxasmParser::COLON);
                  setState(998);
                  test();
                  break;
                }

                case pyxasmParser::POWER: {
                  setState(1000);
                  match(pyxasmParser::POWER);
                  setState(1001);
                  expr();
                  break;
                }

              default:
                throw NoViableAltException(this);
              } 
            }
            setState(1008);
            _errHandler->sync(this);
            alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 145, _ctx);
          }
          setState(1010);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == pyxasmParser::COMMA) {
            setState(1009);
            match(pyxasmParser::COMMA);
          }
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      break;
    }

    case 2: {
      setState(1016);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case pyxasmParser::STRING:
        case pyxasmParser::NUMBER:
        case pyxasmParser::LAMBDA:
        case pyxasmParser::NOT:
        case pyxasmParser::NONE:
        case pyxasmParser::TRUE:
        case pyxasmParser::FALSE:
        case pyxasmParser::AWAIT:
        case pyxasmParser::NAME:
        case pyxasmParser::ELLIPSIS:
        case pyxasmParser::OPEN_PAREN:
        case pyxasmParser::OPEN_BRACK:
        case pyxasmParser::ADD:
        case pyxasmParser::MINUS:
        case pyxasmParser::NOT_OP:
        case pyxasmParser::OPEN_BRACE: {
          setState(1014);
          test();
          break;
        }

        case pyxasmParser::STAR: {
          setState(1015);
          star_expr();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(1032);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case pyxasmParser::FOR:
        case pyxasmParser::ASYNC: {
          setState(1018);
          comp_for();
          break;
        }

        case pyxasmParser::COMMA:
        case pyxasmParser::CLOSE_BRACE: {
          setState(1026);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 150, _ctx);
          while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
            if (alt == 1) {
              setState(1019);
              match(pyxasmParser::COMMA);
              setState(1022);
              _errHandler->sync(this);
              switch (_input->LA(1)) {
                case pyxasmParser::STRING:
                case pyxasmParser::NUMBER:
                case pyxasmParser::LAMBDA:
                case pyxasmParser::NOT:
                case pyxasmParser::NONE:
                case pyxasmParser::TRUE:
                case pyxasmParser::FALSE:
                case pyxasmParser::AWAIT:
                case pyxasmParser::NAME:
                case pyxasmParser::ELLIPSIS:
                case pyxasmParser::OPEN_PAREN:
                case pyxasmParser::OPEN_BRACK:
                case pyxasmParser::ADD:
                case pyxasmParser::MINUS:
                case pyxasmParser::NOT_OP:
                case pyxasmParser::OPEN_BRACE: {
                  setState(1020);
                  test();
                  break;
                }

                case pyxasmParser::STAR: {
                  setState(1021);
                  star_expr();
                  break;
                }

              default:
                throw NoViableAltException(this);
              } 
            }
            setState(1028);
            _errHandler->sync(this);
            alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 150, _ctx);
          }
          setState(1030);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == pyxasmParser::COMMA) {
            setState(1029);
            match(pyxasmParser::COMMA);
          }
          break;
        }

      default:
        throw NoViableAltException(this);
      }
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

//----------------- ClassdefContext ------------------------------------------------------------------

pyxasmParser::ClassdefContext::ClassdefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::ClassdefContext::CLASS() {
  return getToken(pyxasmParser::CLASS, 0);
}

tree::TerminalNode* pyxasmParser::ClassdefContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}

tree::TerminalNode* pyxasmParser::ClassdefContext::COLON() {
  return getToken(pyxasmParser::COLON, 0);
}

pyxasmParser::SuiteContext* pyxasmParser::ClassdefContext::suite() {
  return getRuleContext<pyxasmParser::SuiteContext>(0);
}

tree::TerminalNode* pyxasmParser::ClassdefContext::OPEN_PAREN() {
  return getToken(pyxasmParser::OPEN_PAREN, 0);
}

tree::TerminalNode* pyxasmParser::ClassdefContext::CLOSE_PAREN() {
  return getToken(pyxasmParser::CLOSE_PAREN, 0);
}

pyxasmParser::ArglistContext* pyxasmParser::ClassdefContext::arglist() {
  return getRuleContext<pyxasmParser::ArglistContext>(0);
}


size_t pyxasmParser::ClassdefContext::getRuleIndex() const {
  return pyxasmParser::RuleClassdef;
}

void pyxasmParser::ClassdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassdef(this);
}

void pyxasmParser::ClassdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassdef(this);
}


antlrcpp::Any pyxasmParser::ClassdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitClassdef(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::ClassdefContext* pyxasmParser::classdef() {
  ClassdefContext *_localctx = _tracker.createInstance<ClassdefContext>(_ctx, getState());
  enterRule(_localctx, 154, pyxasmParser::RuleClassdef);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1036);
    match(pyxasmParser::CLASS);
    setState(1037);
    match(pyxasmParser::NAME);
    setState(1043);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::OPEN_PAREN) {
      setState(1038);
      match(pyxasmParser::OPEN_PAREN);
      setState(1040);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
        | (1ULL << pyxasmParser::NUMBER)
        | (1ULL << pyxasmParser::LAMBDA)
        | (1ULL << pyxasmParser::NOT)
        | (1ULL << pyxasmParser::NONE)
        | (1ULL << pyxasmParser::TRUE)
        | (1ULL << pyxasmParser::FALSE)
        | (1ULL << pyxasmParser::AWAIT)
        | (1ULL << pyxasmParser::NAME)
        | (1ULL << pyxasmParser::ELLIPSIS)
        | (1ULL << pyxasmParser::STAR)
        | (1ULL << pyxasmParser::OPEN_PAREN)
        | (1ULL << pyxasmParser::POWER)
        | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
        | (1ULL << (pyxasmParser::MINUS - 66))
        | (1ULL << (pyxasmParser::NOT_OP - 66))
        | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
        setState(1039);
        arglist();
      }
      setState(1042);
      match(pyxasmParser::CLOSE_PAREN);
    }
    setState(1045);
    match(pyxasmParser::COLON);
    setState(1046);
    suite();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArglistContext ------------------------------------------------------------------

pyxasmParser::ArglistContext::ArglistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::ArgumentContext *> pyxasmParser::ArglistContext::argument() {
  return getRuleContexts<pyxasmParser::ArgumentContext>();
}

pyxasmParser::ArgumentContext* pyxasmParser::ArglistContext::argument(size_t i) {
  return getRuleContext<pyxasmParser::ArgumentContext>(i);
}

std::vector<tree::TerminalNode *> pyxasmParser::ArglistContext::COMMA() {
  return getTokens(pyxasmParser::COMMA);
}

tree::TerminalNode* pyxasmParser::ArglistContext::COMMA(size_t i) {
  return getToken(pyxasmParser::COMMA, i);
}


size_t pyxasmParser::ArglistContext::getRuleIndex() const {
  return pyxasmParser::RuleArglist;
}

void pyxasmParser::ArglistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArglist(this);
}

void pyxasmParser::ArglistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArglist(this);
}


antlrcpp::Any pyxasmParser::ArglistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitArglist(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::ArglistContext* pyxasmParser::arglist() {
  ArglistContext *_localctx = _tracker.createInstance<ArglistContext>(_ctx, getState());
  enterRule(_localctx, 156, pyxasmParser::RuleArglist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1048);
    argument();
    setState(1053);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 156, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1049);
        match(pyxasmParser::COMMA);
        setState(1050);
        argument(); 
      }
      setState(1055);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 156, _ctx);
    }
    setState(1057);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::COMMA) {
      setState(1056);
      match(pyxasmParser::COMMA);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArgumentContext ------------------------------------------------------------------

pyxasmParser::ArgumentContext::ArgumentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyxasmParser::TestContext *> pyxasmParser::ArgumentContext::test() {
  return getRuleContexts<pyxasmParser::TestContext>();
}

pyxasmParser::TestContext* pyxasmParser::ArgumentContext::test(size_t i) {
  return getRuleContext<pyxasmParser::TestContext>(i);
}

tree::TerminalNode* pyxasmParser::ArgumentContext::ASSIGN() {
  return getToken(pyxasmParser::ASSIGN, 0);
}

tree::TerminalNode* pyxasmParser::ArgumentContext::POWER() {
  return getToken(pyxasmParser::POWER, 0);
}

tree::TerminalNode* pyxasmParser::ArgumentContext::STAR() {
  return getToken(pyxasmParser::STAR, 0);
}

pyxasmParser::Comp_forContext* pyxasmParser::ArgumentContext::comp_for() {
  return getRuleContext<pyxasmParser::Comp_forContext>(0);
}


size_t pyxasmParser::ArgumentContext::getRuleIndex() const {
  return pyxasmParser::RuleArgument;
}

void pyxasmParser::ArgumentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArgument(this);
}

void pyxasmParser::ArgumentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArgument(this);
}


antlrcpp::Any pyxasmParser::ArgumentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitArgument(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::ArgumentContext* pyxasmParser::argument() {
  ArgumentContext *_localctx = _tracker.createInstance<ArgumentContext>(_ctx, getState());
  enterRule(_localctx, 158, pyxasmParser::RuleArgument);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1071);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 159, _ctx)) {
    case 1: {
      setState(1059);
      test();
      setState(1061);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == pyxasmParser::FOR

      || _la == pyxasmParser::ASYNC) {
        setState(1060);
        comp_for();
      }
      break;
    }

    case 2: {
      setState(1063);
      test();
      setState(1064);
      match(pyxasmParser::ASSIGN);
      setState(1065);
      test();
      break;
    }

    case 3: {
      setState(1067);
      match(pyxasmParser::POWER);
      setState(1068);
      test();
      break;
    }

    case 4: {
      setState(1069);
      match(pyxasmParser::STAR);
      setState(1070);
      test();
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

//----------------- Comp_iterContext ------------------------------------------------------------------

pyxasmParser::Comp_iterContext::Comp_iterContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyxasmParser::Comp_forContext* pyxasmParser::Comp_iterContext::comp_for() {
  return getRuleContext<pyxasmParser::Comp_forContext>(0);
}

pyxasmParser::Comp_ifContext* pyxasmParser::Comp_iterContext::comp_if() {
  return getRuleContext<pyxasmParser::Comp_ifContext>(0);
}


size_t pyxasmParser::Comp_iterContext::getRuleIndex() const {
  return pyxasmParser::RuleComp_iter;
}

void pyxasmParser::Comp_iterContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComp_iter(this);
}

void pyxasmParser::Comp_iterContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComp_iter(this);
}


antlrcpp::Any pyxasmParser::Comp_iterContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitComp_iter(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Comp_iterContext* pyxasmParser::comp_iter() {
  Comp_iterContext *_localctx = _tracker.createInstance<Comp_iterContext>(_ctx, getState());
  enterRule(_localctx, 160, pyxasmParser::RuleComp_iter);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(1075);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::FOR:
      case pyxasmParser::ASYNC: {
        enterOuterAlt(_localctx, 1);
        setState(1073);
        comp_for();
        break;
      }

      case pyxasmParser::IF: {
        enterOuterAlt(_localctx, 2);
        setState(1074);
        comp_if();
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

//----------------- Comp_forContext ------------------------------------------------------------------

pyxasmParser::Comp_forContext::Comp_forContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Comp_forContext::FOR() {
  return getToken(pyxasmParser::FOR, 0);
}

pyxasmParser::ExprlistContext* pyxasmParser::Comp_forContext::exprlist() {
  return getRuleContext<pyxasmParser::ExprlistContext>(0);
}

tree::TerminalNode* pyxasmParser::Comp_forContext::IN() {
  return getToken(pyxasmParser::IN, 0);
}

pyxasmParser::Or_testContext* pyxasmParser::Comp_forContext::or_test() {
  return getRuleContext<pyxasmParser::Or_testContext>(0);
}

tree::TerminalNode* pyxasmParser::Comp_forContext::ASYNC() {
  return getToken(pyxasmParser::ASYNC, 0);
}

pyxasmParser::Comp_iterContext* pyxasmParser::Comp_forContext::comp_iter() {
  return getRuleContext<pyxasmParser::Comp_iterContext>(0);
}


size_t pyxasmParser::Comp_forContext::getRuleIndex() const {
  return pyxasmParser::RuleComp_for;
}

void pyxasmParser::Comp_forContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComp_for(this);
}

void pyxasmParser::Comp_forContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComp_for(this);
}


antlrcpp::Any pyxasmParser::Comp_forContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitComp_for(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Comp_forContext* pyxasmParser::comp_for() {
  Comp_forContext *_localctx = _tracker.createInstance<Comp_forContext>(_ctx, getState());
  enterRule(_localctx, 162, pyxasmParser::RuleComp_for);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1078);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == pyxasmParser::ASYNC) {
      setState(1077);
      match(pyxasmParser::ASYNC);
    }
    setState(1080);
    match(pyxasmParser::FOR);
    setState(1081);
    exprlist();
    setState(1082);
    match(pyxasmParser::IN);
    setState(1083);
    or_test();
    setState(1085);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::IF)
      | (1ULL << pyxasmParser::FOR)
      | (1ULL << pyxasmParser::ASYNC))) != 0)) {
      setState(1084);
      comp_iter();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Comp_ifContext ------------------------------------------------------------------

pyxasmParser::Comp_ifContext::Comp_ifContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Comp_ifContext::IF() {
  return getToken(pyxasmParser::IF, 0);
}

pyxasmParser::Test_nocondContext* pyxasmParser::Comp_ifContext::test_nocond() {
  return getRuleContext<pyxasmParser::Test_nocondContext>(0);
}

pyxasmParser::Comp_iterContext* pyxasmParser::Comp_ifContext::comp_iter() {
  return getRuleContext<pyxasmParser::Comp_iterContext>(0);
}


size_t pyxasmParser::Comp_ifContext::getRuleIndex() const {
  return pyxasmParser::RuleComp_if;
}

void pyxasmParser::Comp_ifContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComp_if(this);
}

void pyxasmParser::Comp_ifContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComp_if(this);
}


antlrcpp::Any pyxasmParser::Comp_ifContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitComp_if(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Comp_ifContext* pyxasmParser::comp_if() {
  Comp_ifContext *_localctx = _tracker.createInstance<Comp_ifContext>(_ctx, getState());
  enterRule(_localctx, 164, pyxasmParser::RuleComp_if);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1087);
    match(pyxasmParser::IF);
    setState(1088);
    test_nocond();
    setState(1090);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::IF)
      | (1ULL << pyxasmParser::FOR)
      | (1ULL << pyxasmParser::ASYNC))) != 0)) {
      setState(1089);
      comp_iter();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Encoding_declContext ------------------------------------------------------------------

pyxasmParser::Encoding_declContext::Encoding_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Encoding_declContext::NAME() {
  return getToken(pyxasmParser::NAME, 0);
}


size_t pyxasmParser::Encoding_declContext::getRuleIndex() const {
  return pyxasmParser::RuleEncoding_decl;
}

void pyxasmParser::Encoding_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEncoding_decl(this);
}

void pyxasmParser::Encoding_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEncoding_decl(this);
}


antlrcpp::Any pyxasmParser::Encoding_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitEncoding_decl(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Encoding_declContext* pyxasmParser::encoding_decl() {
  Encoding_declContext *_localctx = _tracker.createInstance<Encoding_declContext>(_ctx, getState());
  enterRule(_localctx, 166, pyxasmParser::RuleEncoding_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1092);
    match(pyxasmParser::NAME);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Yield_exprContext ------------------------------------------------------------------

pyxasmParser::Yield_exprContext::Yield_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Yield_exprContext::YIELD() {
  return getToken(pyxasmParser::YIELD, 0);
}

pyxasmParser::Yield_argContext* pyxasmParser::Yield_exprContext::yield_arg() {
  return getRuleContext<pyxasmParser::Yield_argContext>(0);
}


size_t pyxasmParser::Yield_exprContext::getRuleIndex() const {
  return pyxasmParser::RuleYield_expr;
}

void pyxasmParser::Yield_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterYield_expr(this);
}

void pyxasmParser::Yield_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitYield_expr(this);
}


antlrcpp::Any pyxasmParser::Yield_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitYield_expr(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Yield_exprContext* pyxasmParser::yield_expr() {
  Yield_exprContext *_localctx = _tracker.createInstance<Yield_exprContext>(_ctx, getState());
  enterRule(_localctx, 168, pyxasmParser::RuleYield_expr);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1094);
    match(pyxasmParser::YIELD);
    setState(1096);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyxasmParser::STRING)
      | (1ULL << pyxasmParser::NUMBER)
      | (1ULL << pyxasmParser::FROM)
      | (1ULL << pyxasmParser::LAMBDA)
      | (1ULL << pyxasmParser::NOT)
      | (1ULL << pyxasmParser::NONE)
      | (1ULL << pyxasmParser::TRUE)
      | (1ULL << pyxasmParser::FALSE)
      | (1ULL << pyxasmParser::AWAIT)
      | (1ULL << pyxasmParser::NAME)
      | (1ULL << pyxasmParser::ELLIPSIS)
      | (1ULL << pyxasmParser::OPEN_PAREN)
      | (1ULL << pyxasmParser::OPEN_BRACK))) != 0) || ((((_la - 66) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 66)) & ((1ULL << (pyxasmParser::ADD - 66))
      | (1ULL << (pyxasmParser::MINUS - 66))
      | (1ULL << (pyxasmParser::NOT_OP - 66))
      | (1ULL << (pyxasmParser::OPEN_BRACE - 66)))) != 0)) {
      setState(1095);
      yield_arg();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Yield_argContext ------------------------------------------------------------------

pyxasmParser::Yield_argContext::Yield_argContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyxasmParser::Yield_argContext::FROM() {
  return getToken(pyxasmParser::FROM, 0);
}

pyxasmParser::TestContext* pyxasmParser::Yield_argContext::test() {
  return getRuleContext<pyxasmParser::TestContext>(0);
}

pyxasmParser::TestlistContext* pyxasmParser::Yield_argContext::testlist() {
  return getRuleContext<pyxasmParser::TestlistContext>(0);
}


size_t pyxasmParser::Yield_argContext::getRuleIndex() const {
  return pyxasmParser::RuleYield_arg;
}

void pyxasmParser::Yield_argContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterYield_arg(this);
}

void pyxasmParser::Yield_argContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyxasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitYield_arg(this);
}


antlrcpp::Any pyxasmParser::Yield_argContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyxasmVisitor*>(visitor))
    return parserVisitor->visitYield_arg(this);
  else
    return visitor->visitChildren(this);
}

pyxasmParser::Yield_argContext* pyxasmParser::yield_arg() {
  Yield_argContext *_localctx = _tracker.createInstance<Yield_argContext>(_ctx, getState());
  enterRule(_localctx, 170, pyxasmParser::RuleYield_arg);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(1101);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyxasmParser::FROM: {
        enterOuterAlt(_localctx, 1);
        setState(1098);
        match(pyxasmParser::FROM);
        setState(1099);
        test();
        break;
      }

      case pyxasmParser::STRING:
      case pyxasmParser::NUMBER:
      case pyxasmParser::LAMBDA:
      case pyxasmParser::NOT:
      case pyxasmParser::NONE:
      case pyxasmParser::TRUE:
      case pyxasmParser::FALSE:
      case pyxasmParser::AWAIT:
      case pyxasmParser::NAME:
      case pyxasmParser::ELLIPSIS:
      case pyxasmParser::OPEN_PAREN:
      case pyxasmParser::OPEN_BRACK:
      case pyxasmParser::ADD:
      case pyxasmParser::MINUS:
      case pyxasmParser::NOT_OP:
      case pyxasmParser::OPEN_BRACE: {
        enterOuterAlt(_localctx, 2);
        setState(1100);
        testlist();
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

// Static vars and initialization.
std::vector<dfa::DFA> pyxasmParser::_decisionToDFA;
atn::PredictionContextCache pyxasmParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN pyxasmParser::_atn;
std::vector<uint16_t> pyxasmParser::_serializedATN;

std::vector<std::string> pyxasmParser::_ruleNames = {
  "single_input", "file_input", "eval_input", "decorator", "decorators", 
  "decorated", "async_funcdef", "funcdef", "parameters", "typedargslist", 
  "tfpdef", "varargslist", "vfpdef", "stmt", "simple_stmt", "small_stmt", 
  "expr_stmt", "annassign", "testlist_star_expr", "augassign", "del_stmt", 
  "pass_stmt", "flow_stmt", "break_stmt", "continue_stmt", "return_stmt", 
  "yield_stmt", "raise_stmt", "import_stmt", "import_name", "import_from", 
  "import_as_name", "dotted_as_name", "import_as_names", "dotted_as_names", 
  "dotted_name", "global_stmt", "nonlocal_stmt", "assert_stmt", "compound_stmt", 
  "async_stmt", "if_stmt", "while_stmt", "for_stmt", "try_stmt", "with_stmt", 
  "with_item", "except_clause", "suite", "test", "test_nocond", "lambdef", 
  "lambdef_nocond", "or_test", "and_test", "not_test", "comparison", "comp_op", 
  "star_expr", "expr", "xor_expr", "and_expr", "shift_expr", "arith_expr", 
  "term", "factor", "power", "atom_expr", "atom", "testlist_comp", "trailer", 
  "subscriptlist", "subscript", "sliceop", "exprlist", "testlist", "dictorsetmaker", 
  "classdef", "arglist", "argument", "comp_iter", "comp_for", "comp_if", 
  "encoding_decl", "yield_expr", "yield_arg"
};

std::vector<std::string> pyxasmParser::_literalNames = {
  "", "", "", "", "'def'", "'return'", "'raise'", "'from'", "'import'", 
  "'as'", "'global'", "'nonlocal'", "'assert'", "'if'", "'elif'", "'else'", 
  "'while'", "'for'", "'in'", "'try'", "'finally'", "'with'", "'except'", 
  "'lambda'", "'or'", "'and'", "'not'", "'is'", "'None'", "'True'", "'False'", 
  "'class'", "'yield'", "'del'", "'pass'", "'continue'", "'break'", "'async'", 
  "'await'", "", "", "", "", "", "", "", "", "", "", "'.'", "'...'", "'*'", 
  "'('", "')'", "','", "':'", "';'", "'**'", "'='", "'['", "']'", "'|'", 
  "'^'", "'&'", "'<<'", "'>>'", "'+'", "'-'", "'/'", "'%'", "'//'", "'~'", 
  "'{'", "'}'", "'<'", "'>'", "'=='", "'>='", "'<='", "'<>'", "'!='", "'@'", 
  "'->'", "'+='", "'-='", "'*='", "'@='", "'/='", "'%='", "'&='", "'|='", 
  "'^='", "'<<='", "'>>='", "'**='", "'//='"
};

std::vector<std::string> pyxasmParser::_symbolicNames = {
  "", "STRING", "NUMBER", "INTEGER", "DEF", "RETURN", "RAISE", "FROM", "IMPORT", 
  "AS", "GLOBAL", "NONLOCAL", "ASSERT", "IF", "ELIF", "ELSE", "WHILE", "FOR", 
  "IN", "TRY", "FINALLY", "WITH", "EXCEPT", "LAMBDA", "OR", "AND", "NOT", 
  "IS", "NONE", "TRUE", "FALSE", "CLASS", "YIELD", "DEL", "PASS", "CONTINUE", 
  "BREAK", "ASYNC", "AWAIT", "NEWLINE", "NAME", "STRING_LITERAL", "BYTES_LITERAL", 
  "DECIMAL_INTEGER", "OCT_INTEGER", "HEX_INTEGER", "BIN_INTEGER", "FLOAT_NUMBER", 
  "IMAG_NUMBER", "DOT", "ELLIPSIS", "STAR", "OPEN_PAREN", "CLOSE_PAREN", 
  "COMMA", "COLON", "SEMI_COLON", "POWER", "ASSIGN", "OPEN_BRACK", "CLOSE_BRACK", 
  "OR_OP", "XOR", "AND_OP", "LEFT_SHIFT", "RIGHT_SHIFT", "ADD", "MINUS", 
  "DIV", "MOD", "IDIV", "NOT_OP", "OPEN_BRACE", "CLOSE_BRACE", "LESS_THAN", 
  "GREATER_THAN", "EQUALS", "GT_EQ", "LT_EQ", "NOT_EQ_1", "NOT_EQ_2", "AT", 
  "ARROW", "ADD_ASSIGN", "SUB_ASSIGN", "MULT_ASSIGN", "AT_ASSIGN", "DIV_ASSIGN", 
  "MOD_ASSIGN", "AND_ASSIGN", "OR_ASSIGN", "XOR_ASSIGN", "LEFT_SHIFT_ASSIGN", 
  "RIGHT_SHIFT_ASSIGN", "POWER_ASSIGN", "IDIV_ASSIGN", "SKIP_", "UNKNOWN_CHAR", 
  "INDENT", "DEDENT"
};

dfa::Vocabulary pyxasmParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> pyxasmParser::_tokenNames;

pyxasmParser::Initializer::Initializer() {
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
    0x3, 0x65, 0x452, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 
    0x1f, 0x9, 0x1f, 0x4, 0x20, 0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x4, 0x22, 
    0x9, 0x22, 0x4, 0x23, 0x9, 0x23, 0x4, 0x24, 0x9, 0x24, 0x4, 0x25, 0x9, 
    0x25, 0x4, 0x26, 0x9, 0x26, 0x4, 0x27, 0x9, 0x27, 0x4, 0x28, 0x9, 0x28, 
    0x4, 0x29, 0x9, 0x29, 0x4, 0x2a, 0x9, 0x2a, 0x4, 0x2b, 0x9, 0x2b, 0x4, 
    0x2c, 0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x4, 0x2e, 0x9, 0x2e, 0x4, 0x2f, 
    0x9, 0x2f, 0x4, 0x30, 0x9, 0x30, 0x4, 0x31, 0x9, 0x31, 0x4, 0x32, 0x9, 
    0x32, 0x4, 0x33, 0x9, 0x33, 0x4, 0x34, 0x9, 0x34, 0x4, 0x35, 0x9, 0x35, 
    0x4, 0x36, 0x9, 0x36, 0x4, 0x37, 0x9, 0x37, 0x4, 0x38, 0x9, 0x38, 0x4, 
    0x39, 0x9, 0x39, 0x4, 0x3a, 0x9, 0x3a, 0x4, 0x3b, 0x9, 0x3b, 0x4, 0x3c, 
    0x9, 0x3c, 0x4, 0x3d, 0x9, 0x3d, 0x4, 0x3e, 0x9, 0x3e, 0x4, 0x3f, 0x9, 
    0x3f, 0x4, 0x40, 0x9, 0x40, 0x4, 0x41, 0x9, 0x41, 0x4, 0x42, 0x9, 0x42, 
    0x4, 0x43, 0x9, 0x43, 0x4, 0x44, 0x9, 0x44, 0x4, 0x45, 0x9, 0x45, 0x4, 
    0x46, 0x9, 0x46, 0x4, 0x47, 0x9, 0x47, 0x4, 0x48, 0x9, 0x48, 0x4, 0x49, 
    0x9, 0x49, 0x4, 0x4a, 0x9, 0x4a, 0x4, 0x4b, 0x9, 0x4b, 0x4, 0x4c, 0x9, 
    0x4c, 0x4, 0x4d, 0x9, 0x4d, 0x4, 0x4e, 0x9, 0x4e, 0x4, 0x4f, 0x9, 0x4f, 
    0x4, 0x50, 0x9, 0x50, 0x4, 0x51, 0x9, 0x51, 0x4, 0x52, 0x9, 0x52, 0x4, 
    0x53, 0x9, 0x53, 0x4, 0x54, 0x9, 0x54, 0x4, 0x55, 0x9, 0x55, 0x4, 0x56, 
    0x9, 0x56, 0x4, 0x57, 0x9, 0x57, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 
    0x2, 0x3, 0x2, 0x5, 0x2, 0xb4, 0xa, 0x2, 0x3, 0x3, 0x3, 0x3, 0x7, 0x3, 
    0xb8, 0xa, 0x3, 0xc, 0x3, 0xe, 0x3, 0xbb, 0xb, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x4, 0x3, 0x4, 0x7, 0x4, 0xc1, 0xa, 0x4, 0xc, 0x4, 0xe, 0x4, 0xc4, 
    0xb, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 
    0x5, 0x5, 0xcc, 0xa, 0x5, 0x3, 0x5, 0x5, 0x5, 0xcf, 0xa, 0x5, 0x3, 0x5, 
    0x3, 0x5, 0x3, 0x6, 0x6, 0x6, 0xd4, 0xa, 0x6, 0xd, 0x6, 0xe, 0x6, 0xd5, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0xdc, 0xa, 0x7, 0x3, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x5, 0x9, 0xe6, 0xa, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 
    0x3, 0xa, 0x5, 0xa, 0xed, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 
    0xb, 0x3, 0xb, 0x5, 0xb, 0xf4, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x5, 0xb, 0xfa, 0xa, 0xb, 0x7, 0xb, 0xfc, 0xa, 0xb, 0xc, 0xb, 
    0xe, 0xb, 0xff, 0xb, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x104, 
    0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x10a, 0xa, 
    0xb, 0x7, 0xb, 0x10c, 0xa, 0xb, 0xc, 0xb, 0xe, 0xb, 0x10f, 0xb, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x115, 0xa, 0xb, 0x5, 
    0xb, 0x117, 0xa, 0xb, 0x5, 0xb, 0x119, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x5, 0xb, 0x11e, 0xa, 0xb, 0x5, 0xb, 0x120, 0xa, 0xb, 0x5, 
    0xb, 0x122, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x126, 0xa, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x12c, 0xa, 0xb, 0x7, 
    0xb, 0x12e, 0xa, 0xb, 0xc, 0xb, 0xe, 0xb, 0x131, 0xb, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x137, 0xa, 0xb, 0x5, 0xb, 0x139, 
    0xa, 0xb, 0x5, 0xb, 0x13b, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 
    0xb, 0x140, 0xa, 0xb, 0x5, 0xb, 0x142, 0xa, 0xb, 0x3, 0xc, 0x3, 0xc, 
    0x3, 0xc, 0x5, 0xc, 0x147, 0xa, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 
    0xd, 0x14c, 0xa, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 
    0x152, 0xa, 0xd, 0x7, 0xd, 0x154, 0xa, 0xd, 0xc, 0xd, 0xe, 0xd, 0x157, 
    0xb, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x15c, 0xa, 0xd, 0x3, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x162, 0xa, 0xd, 0x7, 0xd, 
    0x164, 0xa, 0xd, 0xc, 0xd, 0xe, 0xd, 0x167, 0xb, 0xd, 0x3, 0xd, 0x3, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x16d, 0xa, 0xd, 0x5, 0xd, 0x16f, 
    0xa, 0xd, 0x5, 0xd, 0x171, 0xa, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 
    0xd, 0x176, 0xa, 0xd, 0x5, 0xd, 0x178, 0xa, 0xd, 0x5, 0xd, 0x17a, 0xa, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x17e, 0xa, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x184, 0xa, 0xd, 0x7, 0xd, 0x186, 0xa, 
    0xd, 0xc, 0xd, 0xe, 0xd, 0x189, 0xb, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x5, 0xd, 0x18f, 0xa, 0xd, 0x5, 0xd, 0x191, 0xa, 0xd, 0x5, 
    0xd, 0x193, 0xa, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x198, 
    0xa, 0xd, 0x5, 0xd, 0x19a, 0xa, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 
    0xf, 0x5, 0xf, 0x1a0, 0xa, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x7, 
    0x10, 0x1a5, 0xa, 0x10, 0xc, 0x10, 0xe, 0x10, 0x1a8, 0xb, 0x10, 0x3, 
    0x10, 0x5, 0x10, 0x1ab, 0xa, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x11, 0x5, 0x11, 0x1b7, 0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x3, 0x12, 0x3, 0x12, 0x5, 0x12, 0x1be, 0xa, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x3, 0x12, 0x5, 0x12, 0x1c3, 0xa, 0x12, 0x7, 0x12, 0x1c5, 0xa, 0x12, 
    0xc, 0x12, 0xe, 0x12, 0x1c8, 0xb, 0x12, 0x5, 0x12, 0x1ca, 0xa, 0x12, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x5, 0x13, 0x1d0, 0xa, 0x13, 
    0x3, 0x14, 0x3, 0x14, 0x5, 0x14, 0x1d4, 0xa, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x5, 0x14, 0x1d9, 0xa, 0x14, 0x7, 0x14, 0x1db, 0xa, 0x14, 
    0xc, 0x14, 0xe, 0x14, 0x1de, 0xb, 0x14, 0x3, 0x14, 0x5, 0x14, 0x1e1, 
    0xa, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 
    0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 
    0x5, 0x18, 0x1ef, 0xa, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x1a, 0x3, 0x1a, 
    0x3, 0x1b, 0x3, 0x1b, 0x5, 0x1b, 0x1f7, 0xa, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 
    0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x1ff, 0xa, 0x1d, 
    0x5, 0x1d, 0x201, 0xa, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x5, 0x1e, 0x205, 
    0xa, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x7, 
    0x20, 0x20c, 0xa, 0x20, 0xc, 0x20, 0xe, 0x20, 0x20f, 0xb, 0x20, 0x3, 
    0x20, 0x3, 0x20, 0x6, 0x20, 0x213, 0xa, 0x20, 0xd, 0x20, 0xe, 0x20, 
    0x214, 0x5, 0x20, 0x217, 0xa, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
    0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x5, 0x20, 0x220, 0xa, 0x20, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x5, 0x21, 0x225, 0xa, 0x21, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 0x22a, 0xa, 0x22, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x7, 0x23, 0x22f, 0xa, 0x23, 0xc, 0x23, 0xe, 0x23, 0x232, 
    0xb, 0x23, 0x3, 0x23, 0x5, 0x23, 0x235, 0xa, 0x23, 0x3, 0x24, 0x3, 0x24, 
    0x3, 0x24, 0x7, 0x24, 0x23a, 0xa, 0x24, 0xc, 0x24, 0xe, 0x24, 0x23d, 
    0xb, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x7, 0x25, 0x242, 0xa, 0x25, 
    0xc, 0x25, 0xe, 0x25, 0x245, 0xb, 0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 
    0x3, 0x26, 0x7, 0x26, 0x24b, 0xa, 0x26, 0xc, 0x26, 0xe, 0x26, 0x24e, 
    0xb, 0x26, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 0x254, 
    0xa, 0x27, 0xc, 0x27, 0xe, 0x27, 0x257, 0xb, 0x27, 0x3, 0x28, 0x3, 0x28, 
    0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 0x25d, 0xa, 0x28, 0x3, 0x29, 0x3, 0x29, 
    0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 
    0x29, 0x5, 0x29, 0x268, 0xa, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 
    0x3, 0x2a, 0x5, 0x2a, 0x26e, 0xa, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 
    0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x7, 
    0x2b, 0x279, 0xa, 0x2b, 0xc, 0x2b, 0xe, 0x2b, 0x27c, 0xb, 0x2b, 0x3, 
    0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 0x281, 0xa, 0x2b, 0x3, 0x2c, 
    0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x5, 
    0x2c, 0x28a, 0xa, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 
    0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x5, 0x2d, 0x295, 
    0xa, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 
    0x2e, 0x3, 0x2e, 0x6, 0x2e, 0x29e, 0xa, 0x2e, 0xd, 0x2e, 0xe, 0x2e, 
    0x29f, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x5, 0x2e, 0x2a5, 0xa, 0x2e, 
    0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x5, 0x2e, 0x2aa, 0xa, 0x2e, 0x3, 0x2e, 
    0x3, 0x2e, 0x3, 0x2e, 0x5, 0x2e, 0x2af, 0xa, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 
    0x3, 0x2f, 0x3, 0x2f, 0x7, 0x2f, 0x2b5, 0xa, 0x2f, 0xc, 0x2f, 0xe, 0x2f, 
    0x2b8, 0xb, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x30, 0x3, 0x30, 
    0x3, 0x30, 0x5, 0x30, 0x2c0, 0xa, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x5, 0x31, 0x2c6, 0xa, 0x31, 0x5, 0x31, 0x2c8, 0xa, 0x31, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x6, 0x32, 0x2ce, 0xa, 0x32, 
    0xd, 0x32, 0xe, 0x32, 0x2cf, 0x3, 0x32, 0x3, 0x32, 0x5, 0x32, 0x2d4, 
    0xa, 0x32, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 
    0x33, 0x5, 0x33, 0x2dc, 0xa, 0x33, 0x3, 0x33, 0x5, 0x33, 0x2df, 0xa, 
    0x33, 0x3, 0x34, 0x3, 0x34, 0x5, 0x34, 0x2e3, 0xa, 0x34, 0x3, 0x35, 
    0x3, 0x35, 0x5, 0x35, 0x2e7, 0xa, 0x35, 0x3, 0x35, 0x3, 0x35, 0x3, 0x35, 
    0x3, 0x36, 0x3, 0x36, 0x5, 0x36, 0x2ee, 0xa, 0x36, 0x3, 0x36, 0x3, 0x36, 
    0x3, 0x36, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x7, 0x37, 0x2f6, 0xa, 0x37, 
    0xc, 0x37, 0xe, 0x37, 0x2f9, 0xb, 0x37, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 
    0x7, 0x38, 0x2fe, 0xa, 0x38, 0xc, 0x38, 0xe, 0x38, 0x301, 0xb, 0x38, 
    0x3, 0x39, 0x3, 0x39, 0x3, 0x39, 0x5, 0x39, 0x306, 0xa, 0x39, 0x3, 0x3a, 
    0x3, 0x3a, 0x3, 0x3a, 0x3, 0x3a, 0x7, 0x3a, 0x30c, 0xa, 0x3a, 0xc, 0x3a, 
    0xe, 0x3a, 0x30f, 0xb, 0x3a, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 
    0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 
    0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x5, 0x3b, 0x31e, 0xa, 0x3b, 0x3, 0x3c, 
    0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x7, 0x3d, 0x326, 
    0xa, 0x3d, 0xc, 0x3d, 0xe, 0x3d, 0x329, 0xb, 0x3d, 0x3, 0x3e, 0x3, 0x3e, 
    0x3, 0x3e, 0x7, 0x3e, 0x32e, 0xa, 0x3e, 0xc, 0x3e, 0xe, 0x3e, 0x331, 
    0xb, 0x3e, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 0x7, 0x3f, 0x336, 0xa, 0x3f, 
    0xc, 0x3f, 0xe, 0x3f, 0x339, 0xb, 0x3f, 0x3, 0x40, 0x3, 0x40, 0x3, 0x40, 
    0x7, 0x40, 0x33e, 0xa, 0x40, 0xc, 0x40, 0xe, 0x40, 0x341, 0xb, 0x40, 
    0x3, 0x41, 0x3, 0x41, 0x3, 0x41, 0x7, 0x41, 0x346, 0xa, 0x41, 0xc, 0x41, 
    0xe, 0x41, 0x349, 0xb, 0x41, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x7, 0x42, 
    0x34e, 0xa, 0x42, 0xc, 0x42, 0xe, 0x42, 0x351, 0xb, 0x42, 0x3, 0x43, 
    0x3, 0x43, 0x3, 0x43, 0x5, 0x43, 0x356, 0xa, 0x43, 0x3, 0x44, 0x3, 0x44, 
    0x3, 0x44, 0x5, 0x44, 0x35b, 0xa, 0x44, 0x3, 0x45, 0x5, 0x45, 0x35e, 
    0xa, 0x45, 0x3, 0x45, 0x3, 0x45, 0x7, 0x45, 0x362, 0xa, 0x45, 0xc, 0x45, 
    0xe, 0x45, 0x365, 0xb, 0x45, 0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x5, 0x46, 
    0x36a, 0xa, 0x46, 0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x5, 0x46, 0x36f, 
    0xa, 0x46, 0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x5, 0x46, 0x374, 0xa, 0x46, 
    0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x6, 0x46, 0x37a, 0xa, 0x46, 
    0xd, 0x46, 0xe, 0x46, 0x37b, 0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 
    0x5, 0x46, 0x382, 0xa, 0x46, 0x3, 0x47, 0x3, 0x47, 0x5, 0x47, 0x386, 
    0xa, 0x47, 0x3, 0x47, 0x3, 0x47, 0x3, 0x47, 0x3, 0x47, 0x5, 0x47, 0x38c, 
    0xa, 0x47, 0x7, 0x47, 0x38e, 0xa, 0x47, 0xc, 0x47, 0xe, 0x47, 0x391, 
    0xb, 0x47, 0x3, 0x47, 0x5, 0x47, 0x394, 0xa, 0x47, 0x5, 0x47, 0x396, 
    0xa, 0x47, 0x3, 0x48, 0x3, 0x48, 0x5, 0x48, 0x39a, 0xa, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x5, 
    0x48, 0x3a3, 0xa, 0x48, 0x3, 0x49, 0x3, 0x49, 0x3, 0x49, 0x7, 0x49, 
    0x3a8, 0xa, 0x49, 0xc, 0x49, 0xe, 0x49, 0x3ab, 0xb, 0x49, 0x3, 0x49, 
    0x5, 0x49, 0x3ae, 0xa, 0x49, 0x3, 0x4a, 0x3, 0x4a, 0x5, 0x4a, 0x3b2, 
    0xa, 0x4a, 0x3, 0x4a, 0x3, 0x4a, 0x5, 0x4a, 0x3b6, 0xa, 0x4a, 0x3, 0x4a, 
    0x5, 0x4a, 0x3b9, 0xa, 0x4a, 0x5, 0x4a, 0x3bb, 0xa, 0x4a, 0x3, 0x4b, 
    0x3, 0x4b, 0x5, 0x4b, 0x3bf, 0xa, 0x4b, 0x3, 0x4c, 0x3, 0x4c, 0x5, 0x4c, 
    0x3c3, 0xa, 0x4c, 0x3, 0x4c, 0x3, 0x4c, 0x3, 0x4c, 0x5, 0x4c, 0x3c8, 
    0xa, 0x4c, 0x7, 0x4c, 0x3ca, 0xa, 0x4c, 0xc, 0x4c, 0xe, 0x4c, 0x3cd, 
    0xb, 0x4c, 0x3, 0x4c, 0x5, 0x4c, 0x3d0, 0xa, 0x4c, 0x3, 0x4d, 0x3, 0x4d, 
    0x3, 0x4d, 0x7, 0x4d, 0x3d5, 0xa, 0x4d, 0xc, 0x4d, 0xe, 0x4d, 0x3d8, 
    0xb, 0x4d, 0x3, 0x4d, 0x5, 0x4d, 0x3db, 0xa, 0x4d, 0x3, 0x4e, 0x3, 0x4e, 
    0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x5, 0x4e, 0x3e3, 0xa, 0x4e, 
    0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 
    0x4e, 0x3, 0x4e, 0x5, 0x4e, 0x3ed, 0xa, 0x4e, 0x7, 0x4e, 0x3ef, 0xa, 
    0x4e, 0xc, 0x4e, 0xe, 0x4e, 0x3f2, 0xb, 0x4e, 0x3, 0x4e, 0x5, 0x4e, 
    0x3f5, 0xa, 0x4e, 0x5, 0x4e, 0x3f7, 0xa, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 
    0x5, 0x4e, 0x3fb, 0xa, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 
    0x5, 0x4e, 0x401, 0xa, 0x4e, 0x7, 0x4e, 0x403, 0xa, 0x4e, 0xc, 0x4e, 
    0xe, 0x4e, 0x406, 0xb, 0x4e, 0x3, 0x4e, 0x5, 0x4e, 0x409, 0xa, 0x4e, 
    0x5, 0x4e, 0x40b, 0xa, 0x4e, 0x5, 0x4e, 0x40d, 0xa, 0x4e, 0x3, 0x4f, 
    0x3, 0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x5, 0x4f, 0x413, 0xa, 0x4f, 0x3, 0x4f, 
    0x5, 0x4f, 0x416, 0xa, 0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x3, 0x50, 
    0x3, 0x50, 0x3, 0x50, 0x7, 0x50, 0x41e, 0xa, 0x50, 0xc, 0x50, 0xe, 0x50, 
    0x421, 0xb, 0x50, 0x3, 0x50, 0x5, 0x50, 0x424, 0xa, 0x50, 0x3, 0x51, 
    0x3, 0x51, 0x5, 0x51, 0x428, 0xa, 0x51, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 
    0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x5, 0x51, 0x432, 
    0xa, 0x51, 0x3, 0x52, 0x3, 0x52, 0x5, 0x52, 0x436, 0xa, 0x52, 0x3, 0x53, 
    0x5, 0x53, 0x439, 0xa, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x53, 
    0x3, 0x53, 0x5, 0x53, 0x440, 0xa, 0x53, 0x3, 0x54, 0x3, 0x54, 0x3, 0x54, 
    0x5, 0x54, 0x445, 0xa, 0x54, 0x3, 0x55, 0x3, 0x55, 0x3, 0x56, 0x3, 0x56, 
    0x5, 0x56, 0x44b, 0xa, 0x56, 0x3, 0x57, 0x3, 0x57, 0x3, 0x57, 0x5, 0x57, 
    0x450, 0xa, 0x57, 0x3, 0x57, 0x2, 0x2, 0x58, 0x2, 0x4, 0x6, 0x8, 0xa, 
    0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 
    0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 
    0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 
    0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 
    0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e, 0x80, 0x82, 
    0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 
    0x9c, 0x9e, 0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0x2, 0x8, 0x3, 
    0x2, 0x55, 0x61, 0x3, 0x2, 0x33, 0x34, 0x3, 0x2, 0x42, 0x43, 0x3, 0x2, 
    0x44, 0x45, 0x5, 0x2, 0x35, 0x35, 0x46, 0x48, 0x53, 0x53, 0x4, 0x2, 
    0x44, 0x45, 0x49, 0x49, 0x2, 0x4cf, 0x2, 0xb3, 0x3, 0x2, 0x2, 0x2, 0x4, 
    0xb9, 0x3, 0x2, 0x2, 0x2, 0x6, 0xbe, 0x3, 0x2, 0x2, 0x2, 0x8, 0xc7, 
    0x3, 0x2, 0x2, 0x2, 0xa, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xc, 0xd7, 0x3, 0x2, 
    0x2, 0x2, 0xe, 0xdd, 0x3, 0x2, 0x2, 0x2, 0x10, 0xe0, 0x3, 0x2, 0x2, 
    0x2, 0x12, 0xea, 0x3, 0x2, 0x2, 0x2, 0x14, 0x141, 0x3, 0x2, 0x2, 0x2, 
    0x16, 0x143, 0x3, 0x2, 0x2, 0x2, 0x18, 0x199, 0x3, 0x2, 0x2, 0x2, 0x1a, 
    0x19b, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x19f, 0x3, 0x2, 0x2, 0x2, 0x1e, 0x1a1, 
    0x3, 0x2, 0x2, 0x2, 0x20, 0x1b6, 0x3, 0x2, 0x2, 0x2, 0x22, 0x1b8, 0x3, 
    0x2, 0x2, 0x2, 0x24, 0x1cb, 0x3, 0x2, 0x2, 0x2, 0x26, 0x1d3, 0x3, 0x2, 
    0x2, 0x2, 0x28, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x2a, 0x1e4, 0x3, 0x2, 0x2, 
    0x2, 0x2c, 0x1e7, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x1ee, 0x3, 0x2, 0x2, 0x2, 
    0x30, 0x1f0, 0x3, 0x2, 0x2, 0x2, 0x32, 0x1f2, 0x3, 0x2, 0x2, 0x2, 0x34, 
    0x1f4, 0x3, 0x2, 0x2, 0x2, 0x36, 0x1f8, 0x3, 0x2, 0x2, 0x2, 0x38, 0x1fa, 
    0x3, 0x2, 0x2, 0x2, 0x3a, 0x204, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x206, 0x3, 
    0x2, 0x2, 0x2, 0x3e, 0x209, 0x3, 0x2, 0x2, 0x2, 0x40, 0x221, 0x3, 0x2, 
    0x2, 0x2, 0x42, 0x226, 0x3, 0x2, 0x2, 0x2, 0x44, 0x22b, 0x3, 0x2, 0x2, 
    0x2, 0x46, 0x236, 0x3, 0x2, 0x2, 0x2, 0x48, 0x23e, 0x3, 0x2, 0x2, 0x2, 
    0x4a, 0x246, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x24f, 0x3, 0x2, 0x2, 0x2, 0x4e, 
    0x258, 0x3, 0x2, 0x2, 0x2, 0x50, 0x267, 0x3, 0x2, 0x2, 0x2, 0x52, 0x269, 
    0x3, 0x2, 0x2, 0x2, 0x54, 0x26f, 0x3, 0x2, 0x2, 0x2, 0x56, 0x282, 0x3, 
    0x2, 0x2, 0x2, 0x58, 0x28b, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x296, 0x3, 0x2, 
    0x2, 0x2, 0x5c, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x2bc, 0x3, 0x2, 0x2, 
    0x2, 0x60, 0x2c1, 0x3, 0x2, 0x2, 0x2, 0x62, 0x2d3, 0x3, 0x2, 0x2, 0x2, 
    0x64, 0x2de, 0x3, 0x2, 0x2, 0x2, 0x66, 0x2e2, 0x3, 0x2, 0x2, 0x2, 0x68, 
    0x2e4, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x2eb, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x2f2, 
    0x3, 0x2, 0x2, 0x2, 0x6e, 0x2fa, 0x3, 0x2, 0x2, 0x2, 0x70, 0x305, 0x3, 
    0x2, 0x2, 0x2, 0x72, 0x307, 0x3, 0x2, 0x2, 0x2, 0x74, 0x31d, 0x3, 0x2, 
    0x2, 0x2, 0x76, 0x31f, 0x3, 0x2, 0x2, 0x2, 0x78, 0x322, 0x3, 0x2, 0x2, 
    0x2, 0x7a, 0x32a, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x332, 0x3, 0x2, 0x2, 0x2, 
    0x7e, 0x33a, 0x3, 0x2, 0x2, 0x2, 0x80, 0x342, 0x3, 0x2, 0x2, 0x2, 0x82, 
    0x34a, 0x3, 0x2, 0x2, 0x2, 0x84, 0x355, 0x3, 0x2, 0x2, 0x2, 0x86, 0x357, 
    0x3, 0x2, 0x2, 0x2, 0x88, 0x35d, 0x3, 0x2, 0x2, 0x2, 0x8a, 0x381, 0x3, 
    0x2, 0x2, 0x2, 0x8c, 0x385, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x3a2, 0x3, 0x2, 
    0x2, 0x2, 0x90, 0x3a4, 0x3, 0x2, 0x2, 0x2, 0x92, 0x3ba, 0x3, 0x2, 0x2, 
    0x2, 0x94, 0x3bc, 0x3, 0x2, 0x2, 0x2, 0x96, 0x3c2, 0x3, 0x2, 0x2, 0x2, 
    0x98, 0x3d1, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x40c, 0x3, 0x2, 0x2, 0x2, 0x9c, 
    0x40e, 0x3, 0x2, 0x2, 0x2, 0x9e, 0x41a, 0x3, 0x2, 0x2, 0x2, 0xa0, 0x431, 
    0x3, 0x2, 0x2, 0x2, 0xa2, 0x435, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x438, 0x3, 
    0x2, 0x2, 0x2, 0xa6, 0x441, 0x3, 0x2, 0x2, 0x2, 0xa8, 0x446, 0x3, 0x2, 
    0x2, 0x2, 0xaa, 0x448, 0x3, 0x2, 0x2, 0x2, 0xac, 0x44f, 0x3, 0x2, 0x2, 
    0x2, 0xae, 0xb4, 0x7, 0x29, 0x2, 0x2, 0xaf, 0xb4, 0x5, 0x1e, 0x10, 0x2, 
    0xb0, 0xb1, 0x5, 0x50, 0x29, 0x2, 0xb1, 0xb2, 0x7, 0x29, 0x2, 0x2, 0xb2, 
    0xb4, 0x3, 0x2, 0x2, 0x2, 0xb3, 0xae, 0x3, 0x2, 0x2, 0x2, 0xb3, 0xaf, 
    0x3, 0x2, 0x2, 0x2, 0xb3, 0xb0, 0x3, 0x2, 0x2, 0x2, 0xb4, 0x3, 0x3, 
    0x2, 0x2, 0x2, 0xb5, 0xb8, 0x7, 0x29, 0x2, 0x2, 0xb6, 0xb8, 0x5, 0x1c, 
    0xf, 0x2, 0xb7, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb7, 0xb6, 0x3, 0x2, 0x2, 
    0x2, 0xb8, 0xbb, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xb7, 0x3, 0x2, 0x2, 0x2, 
    0xb9, 0xba, 0x3, 0x2, 0x2, 0x2, 0xba, 0xbc, 0x3, 0x2, 0x2, 0x2, 0xbb, 
    0xb9, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xbd, 0x7, 0x2, 0x2, 0x3, 0xbd, 0x5, 
    0x3, 0x2, 0x2, 0x2, 0xbe, 0xc2, 0x5, 0x98, 0x4d, 0x2, 0xbf, 0xc1, 0x7, 
    0x29, 0x2, 0x2, 0xc0, 0xbf, 0x3, 0x2, 0x2, 0x2, 0xc1, 0xc4, 0x3, 0x2, 
    0x2, 0x2, 0xc2, 0xc0, 0x3, 0x2, 0x2, 0x2, 0xc2, 0xc3, 0x3, 0x2, 0x2, 
    0x2, 0xc3, 0xc5, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc2, 0x3, 0x2, 0x2, 0x2, 
    0xc5, 0xc6, 0x7, 0x2, 0x2, 0x3, 0xc6, 0x7, 0x3, 0x2, 0x2, 0x2, 0xc7, 
    0xc8, 0x7, 0x53, 0x2, 0x2, 0xc8, 0xce, 0x5, 0x48, 0x25, 0x2, 0xc9, 0xcb, 
    0x7, 0x36, 0x2, 0x2, 0xca, 0xcc, 0x5, 0x9e, 0x50, 0x2, 0xcb, 0xca, 0x3, 
    0x2, 0x2, 0x2, 0xcb, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xcc, 0xcd, 0x3, 0x2, 
    0x2, 0x2, 0xcd, 0xcf, 0x7, 0x37, 0x2, 0x2, 0xce, 0xc9, 0x3, 0x2, 0x2, 
    0x2, 0xce, 0xcf, 0x3, 0x2, 0x2, 0x2, 0xcf, 0xd0, 0x3, 0x2, 0x2, 0x2, 
    0xd0, 0xd1, 0x7, 0x29, 0x2, 0x2, 0xd1, 0x9, 0x3, 0x2, 0x2, 0x2, 0xd2, 
    0xd4, 0x5, 0x8, 0x5, 0x2, 0xd3, 0xd2, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd5, 
    0x3, 0x2, 0x2, 0x2, 0xd5, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd6, 0x3, 
    0x2, 0x2, 0x2, 0xd6, 0xb, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xdb, 0x5, 0xa, 
    0x6, 0x2, 0xd8, 0xdc, 0x5, 0x9c, 0x4f, 0x2, 0xd9, 0xdc, 0x5, 0x10, 0x9, 
    0x2, 0xda, 0xdc, 0x5, 0xe, 0x8, 0x2, 0xdb, 0xd8, 0x3, 0x2, 0x2, 0x2, 
    0xdb, 0xd9, 0x3, 0x2, 0x2, 0x2, 0xdb, 0xda, 0x3, 0x2, 0x2, 0x2, 0xdc, 
    0xd, 0x3, 0x2, 0x2, 0x2, 0xdd, 0xde, 0x7, 0x27, 0x2, 0x2, 0xde, 0xdf, 
    0x5, 0x10, 0x9, 0x2, 0xdf, 0xf, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xe1, 0x7, 
    0x6, 0x2, 0x2, 0xe1, 0xe2, 0x7, 0x2a, 0x2, 0x2, 0xe2, 0xe5, 0x5, 0x12, 
    0xa, 0x2, 0xe3, 0xe4, 0x7, 0x54, 0x2, 0x2, 0xe4, 0xe6, 0x5, 0x64, 0x33, 
    0x2, 0xe5, 0xe3, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe6, 0x3, 0x2, 0x2, 0x2, 
    0xe6, 0xe7, 0x3, 0x2, 0x2, 0x2, 0xe7, 0xe8, 0x7, 0x39, 0x2, 0x2, 0xe8, 
    0xe9, 0x5, 0x62, 0x32, 0x2, 0xe9, 0x11, 0x3, 0x2, 0x2, 0x2, 0xea, 0xec, 
    0x7, 0x36, 0x2, 0x2, 0xeb, 0xed, 0x5, 0x14, 0xb, 0x2, 0xec, 0xeb, 0x3, 
    0x2, 0x2, 0x2, 0xec, 0xed, 0x3, 0x2, 0x2, 0x2, 0xed, 0xee, 0x3, 0x2, 
    0x2, 0x2, 0xee, 0xef, 0x7, 0x37, 0x2, 0x2, 0xef, 0x13, 0x3, 0x2, 0x2, 
    0x2, 0xf0, 0xf3, 0x5, 0x16, 0xc, 0x2, 0xf1, 0xf2, 0x7, 0x3c, 0x2, 0x2, 
    0xf2, 0xf4, 0x5, 0x64, 0x33, 0x2, 0xf3, 0xf1, 0x3, 0x2, 0x2, 0x2, 0xf3, 
    0xf4, 0x3, 0x2, 0x2, 0x2, 0xf4, 0xfd, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xf6, 
    0x7, 0x38, 0x2, 0x2, 0xf6, 0xf9, 0x5, 0x16, 0xc, 0x2, 0xf7, 0xf8, 0x7, 
    0x3c, 0x2, 0x2, 0xf8, 0xfa, 0x5, 0x64, 0x33, 0x2, 0xf9, 0xf7, 0x3, 0x2, 
    0x2, 0x2, 0xf9, 0xfa, 0x3, 0x2, 0x2, 0x2, 0xfa, 0xfc, 0x3, 0x2, 0x2, 
    0x2, 0xfb, 0xf5, 0x3, 0x2, 0x2, 0x2, 0xfc, 0xff, 0x3, 0x2, 0x2, 0x2, 
    0xfd, 0xfb, 0x3, 0x2, 0x2, 0x2, 0xfd, 0xfe, 0x3, 0x2, 0x2, 0x2, 0xfe, 
    0x121, 0x3, 0x2, 0x2, 0x2, 0xff, 0xfd, 0x3, 0x2, 0x2, 0x2, 0x100, 0x11f, 
    0x7, 0x38, 0x2, 0x2, 0x101, 0x103, 0x7, 0x35, 0x2, 0x2, 0x102, 0x104, 
    0x5, 0x16, 0xc, 0x2, 0x103, 0x102, 0x3, 0x2, 0x2, 0x2, 0x103, 0x104, 
    0x3, 0x2, 0x2, 0x2, 0x104, 0x10d, 0x3, 0x2, 0x2, 0x2, 0x105, 0x106, 
    0x7, 0x38, 0x2, 0x2, 0x106, 0x109, 0x5, 0x16, 0xc, 0x2, 0x107, 0x108, 
    0x7, 0x3c, 0x2, 0x2, 0x108, 0x10a, 0x5, 0x64, 0x33, 0x2, 0x109, 0x107, 
    0x3, 0x2, 0x2, 0x2, 0x109, 0x10a, 0x3, 0x2, 0x2, 0x2, 0x10a, 0x10c, 
    0x3, 0x2, 0x2, 0x2, 0x10b, 0x105, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10f, 
    0x3, 0x2, 0x2, 0x2, 0x10d, 0x10b, 0x3, 0x2, 0x2, 0x2, 0x10d, 0x10e, 
    0x3, 0x2, 0x2, 0x2, 0x10e, 0x118, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x10d, 
    0x3, 0x2, 0x2, 0x2, 0x110, 0x116, 0x7, 0x38, 0x2, 0x2, 0x111, 0x112, 
    0x7, 0x3b, 0x2, 0x2, 0x112, 0x114, 0x5, 0x16, 0xc, 0x2, 0x113, 0x115, 
    0x7, 0x38, 0x2, 0x2, 0x114, 0x113, 0x3, 0x2, 0x2, 0x2, 0x114, 0x115, 
    0x3, 0x2, 0x2, 0x2, 0x115, 0x117, 0x3, 0x2, 0x2, 0x2, 0x116, 0x111, 
    0x3, 0x2, 0x2, 0x2, 0x116, 0x117, 0x3, 0x2, 0x2, 0x2, 0x117, 0x119, 
    0x3, 0x2, 0x2, 0x2, 0x118, 0x110, 0x3, 0x2, 0x2, 0x2, 0x118, 0x119, 
    0x3, 0x2, 0x2, 0x2, 0x119, 0x120, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x11b, 
    0x7, 0x3b, 0x2, 0x2, 0x11b, 0x11d, 0x5, 0x16, 0xc, 0x2, 0x11c, 0x11e, 
    0x7, 0x38, 0x2, 0x2, 0x11d, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x11d, 0x11e, 
    0x3, 0x2, 0x2, 0x2, 0x11e, 0x120, 0x3, 0x2, 0x2, 0x2, 0x11f, 0x101, 
    0x3, 0x2, 0x2, 0x2, 0x11f, 0x11a, 0x3, 0x2, 0x2, 0x2, 0x11f, 0x120, 
    0x3, 0x2, 0x2, 0x2, 0x120, 0x122, 0x3, 0x2, 0x2, 0x2, 0x121, 0x100, 
    0x3, 0x2, 0x2, 0x2, 0x121, 0x122, 0x3, 0x2, 0x2, 0x2, 0x122, 0x142, 
    0x3, 0x2, 0x2, 0x2, 0x123, 0x125, 0x7, 0x35, 0x2, 0x2, 0x124, 0x126, 
    0x5, 0x16, 0xc, 0x2, 0x125, 0x124, 0x3, 0x2, 0x2, 0x2, 0x125, 0x126, 
    0x3, 0x2, 0x2, 0x2, 0x126, 0x12f, 0x3, 0x2, 0x2, 0x2, 0x127, 0x128, 
    0x7, 0x38, 0x2, 0x2, 0x128, 0x12b, 0x5, 0x16, 0xc, 0x2, 0x129, 0x12a, 
    0x7, 0x3c, 0x2, 0x2, 0x12a, 0x12c, 0x5, 0x64, 0x33, 0x2, 0x12b, 0x129, 
    0x3, 0x2, 0x2, 0x2, 0x12b, 0x12c, 0x3, 0x2, 0x2, 0x2, 0x12c, 0x12e, 
    0x3, 0x2, 0x2, 0x2, 0x12d, 0x127, 0x3, 0x2, 0x2, 0x2, 0x12e, 0x131, 
    0x3, 0x2, 0x2, 0x2, 0x12f, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x12f, 0x130, 
    0x3, 0x2, 0x2, 0x2, 0x130, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x131, 0x12f, 
    0x3, 0x2, 0x2, 0x2, 0x132, 0x138, 0x7, 0x38, 0x2, 0x2, 0x133, 0x134, 
    0x7, 0x3b, 0x2, 0x2, 0x134, 0x136, 0x5, 0x16, 0xc, 0x2, 0x135, 0x137, 
    0x7, 0x38, 0x2, 0x2, 0x136, 0x135, 0x3, 0x2, 0x2, 0x2, 0x136, 0x137, 
    0x3, 0x2, 0x2, 0x2, 0x137, 0x139, 0x3, 0x2, 0x2, 0x2, 0x138, 0x133, 
    0x3, 0x2, 0x2, 0x2, 0x138, 0x139, 0x3, 0x2, 0x2, 0x2, 0x139, 0x13b, 
    0x3, 0x2, 0x2, 0x2, 0x13a, 0x132, 0x3, 0x2, 0x2, 0x2, 0x13a, 0x13b, 
    0x3, 0x2, 0x2, 0x2, 0x13b, 0x142, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x13d, 
    0x7, 0x3b, 0x2, 0x2, 0x13d, 0x13f, 0x5, 0x16, 0xc, 0x2, 0x13e, 0x140, 
    0x7, 0x38, 0x2, 0x2, 0x13f, 0x13e, 0x3, 0x2, 0x2, 0x2, 0x13f, 0x140, 
    0x3, 0x2, 0x2, 0x2, 0x140, 0x142, 0x3, 0x2, 0x2, 0x2, 0x141, 0xf0, 0x3, 
    0x2, 0x2, 0x2, 0x141, 0x123, 0x3, 0x2, 0x2, 0x2, 0x141, 0x13c, 0x3, 
    0x2, 0x2, 0x2, 0x142, 0x15, 0x3, 0x2, 0x2, 0x2, 0x143, 0x146, 0x7, 0x2a, 
    0x2, 0x2, 0x144, 0x145, 0x7, 0x39, 0x2, 0x2, 0x145, 0x147, 0x5, 0x64, 
    0x33, 0x2, 0x146, 0x144, 0x3, 0x2, 0x2, 0x2, 0x146, 0x147, 0x3, 0x2, 
    0x2, 0x2, 0x147, 0x17, 0x3, 0x2, 0x2, 0x2, 0x148, 0x14b, 0x5, 0x1a, 
    0xe, 0x2, 0x149, 0x14a, 0x7, 0x3c, 0x2, 0x2, 0x14a, 0x14c, 0x5, 0x64, 
    0x33, 0x2, 0x14b, 0x149, 0x3, 0x2, 0x2, 0x2, 0x14b, 0x14c, 0x3, 0x2, 
    0x2, 0x2, 0x14c, 0x155, 0x3, 0x2, 0x2, 0x2, 0x14d, 0x14e, 0x7, 0x38, 
    0x2, 0x2, 0x14e, 0x151, 0x5, 0x1a, 0xe, 0x2, 0x14f, 0x150, 0x7, 0x3c, 
    0x2, 0x2, 0x150, 0x152, 0x5, 0x64, 0x33, 0x2, 0x151, 0x14f, 0x3, 0x2, 
    0x2, 0x2, 0x151, 0x152, 0x3, 0x2, 0x2, 0x2, 0x152, 0x154, 0x3, 0x2, 
    0x2, 0x2, 0x153, 0x14d, 0x3, 0x2, 0x2, 0x2, 0x154, 0x157, 0x3, 0x2, 
    0x2, 0x2, 0x155, 0x153, 0x3, 0x2, 0x2, 0x2, 0x155, 0x156, 0x3, 0x2, 
    0x2, 0x2, 0x156, 0x179, 0x3, 0x2, 0x2, 0x2, 0x157, 0x155, 0x3, 0x2, 
    0x2, 0x2, 0x158, 0x177, 0x7, 0x38, 0x2, 0x2, 0x159, 0x15b, 0x7, 0x35, 
    0x2, 0x2, 0x15a, 0x15c, 0x5, 0x1a, 0xe, 0x2, 0x15b, 0x15a, 0x3, 0x2, 
    0x2, 0x2, 0x15b, 0x15c, 0x3, 0x2, 0x2, 0x2, 0x15c, 0x165, 0x3, 0x2, 
    0x2, 0x2, 0x15d, 0x15e, 0x7, 0x38, 0x2, 0x2, 0x15e, 0x161, 0x5, 0x1a, 
    0xe, 0x2, 0x15f, 0x160, 0x7, 0x3c, 0x2, 0x2, 0x160, 0x162, 0x5, 0x64, 
    0x33, 0x2, 0x161, 0x15f, 0x3, 0x2, 0x2, 0x2, 0x161, 0x162, 0x3, 0x2, 
    0x2, 0x2, 0x162, 0x164, 0x3, 0x2, 0x2, 0x2, 0x163, 0x15d, 0x3, 0x2, 
    0x2, 0x2, 0x164, 0x167, 0x3, 0x2, 0x2, 0x2, 0x165, 0x163, 0x3, 0x2, 
    0x2, 0x2, 0x165, 0x166, 0x3, 0x2, 0x2, 0x2, 0x166, 0x170, 0x3, 0x2, 
    0x2, 0x2, 0x167, 0x165, 0x3, 0x2, 0x2, 0x2, 0x168, 0x16e, 0x7, 0x38, 
    0x2, 0x2, 0x169, 0x16a, 0x7, 0x3b, 0x2, 0x2, 0x16a, 0x16c, 0x5, 0x1a, 
    0xe, 0x2, 0x16b, 0x16d, 0x7, 0x38, 0x2, 0x2, 0x16c, 0x16b, 0x3, 0x2, 
    0x2, 0x2, 0x16c, 0x16d, 0x3, 0x2, 0x2, 0x2, 0x16d, 0x16f, 0x3, 0x2, 
    0x2, 0x2, 0x16e, 0x169, 0x3, 0x2, 0x2, 0x2, 0x16e, 0x16f, 0x3, 0x2, 
    0x2, 0x2, 0x16f, 0x171, 0x3, 0x2, 0x2, 0x2, 0x170, 0x168, 0x3, 0x2, 
    0x2, 0x2, 0x170, 0x171, 0x3, 0x2, 0x2, 0x2, 0x171, 0x178, 0x3, 0x2, 
    0x2, 0x2, 0x172, 0x173, 0x7, 0x3b, 0x2, 0x2, 0x173, 0x175, 0x5, 0x1a, 
    0xe, 0x2, 0x174, 0x176, 0x7, 0x38, 0x2, 0x2, 0x175, 0x174, 0x3, 0x2, 
    0x2, 0x2, 0x175, 0x176, 0x3, 0x2, 0x2, 0x2, 0x176, 0x178, 0x3, 0x2, 
    0x2, 0x2, 0x177, 0x159, 0x3, 0x2, 0x2, 0x2, 0x177, 0x172, 0x3, 0x2, 
    0x2, 0x2, 0x177, 0x178, 0x3, 0x2, 0x2, 0x2, 0x178, 0x17a, 0x3, 0x2, 
    0x2, 0x2, 0x179, 0x158, 0x3, 0x2, 0x2, 0x2, 0x179, 0x17a, 0x3, 0x2, 
    0x2, 0x2, 0x17a, 0x19a, 0x3, 0x2, 0x2, 0x2, 0x17b, 0x17d, 0x7, 0x35, 
    0x2, 0x2, 0x17c, 0x17e, 0x5, 0x1a, 0xe, 0x2, 0x17d, 0x17c, 0x3, 0x2, 
    0x2, 0x2, 0x17d, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x17e, 0x187, 0x3, 0x2, 
    0x2, 0x2, 0x17f, 0x180, 0x7, 0x38, 0x2, 0x2, 0x180, 0x183, 0x5, 0x1a, 
    0xe, 0x2, 0x181, 0x182, 0x7, 0x3c, 0x2, 0x2, 0x182, 0x184, 0x5, 0x64, 
    0x33, 0x2, 0x183, 0x181, 0x3, 0x2, 0x2, 0x2, 0x183, 0x184, 0x3, 0x2, 
    0x2, 0x2, 0x184, 0x186, 0x3, 0x2, 0x2, 0x2, 0x185, 0x17f, 0x3, 0x2, 
    0x2, 0x2, 0x186, 0x189, 0x3, 0x2, 0x2, 0x2, 0x187, 0x185, 0x3, 0x2, 
    0x2, 0x2, 0x187, 0x188, 0x3, 0x2, 0x2, 0x2, 0x188, 0x192, 0x3, 0x2, 
    0x2, 0x2, 0x189, 0x187, 0x3, 0x2, 0x2, 0x2, 0x18a, 0x190, 0x7, 0x38, 
    0x2, 0x2, 0x18b, 0x18c, 0x7, 0x3b, 0x2, 0x2, 0x18c, 0x18e, 0x5, 0x1a, 
    0xe, 0x2, 0x18d, 0x18f, 0x7, 0x38, 0x2, 0x2, 0x18e, 0x18d, 0x3, 0x2, 
    0x2, 0x2, 0x18e, 0x18f, 0x3, 0x2, 0x2, 0x2, 0x18f, 0x191, 0x3, 0x2, 
    0x2, 0x2, 0x190, 0x18b, 0x3, 0x2, 0x2, 0x2, 0x190, 0x191, 0x3, 0x2, 
    0x2, 0x2, 0x191, 0x193, 0x3, 0x2, 0x2, 0x2, 0x192, 0x18a, 0x3, 0x2, 
    0x2, 0x2, 0x192, 0x193, 0x3, 0x2, 0x2, 0x2, 0x193, 0x19a, 0x3, 0x2, 
    0x2, 0x2, 0x194, 0x195, 0x7, 0x3b, 0x2, 0x2, 0x195, 0x197, 0x5, 0x1a, 
    0xe, 0x2, 0x196, 0x198, 0x7, 0x38, 0x2, 0x2, 0x197, 0x196, 0x3, 0x2, 
    0x2, 0x2, 0x197, 0x198, 0x3, 0x2, 0x2, 0x2, 0x198, 0x19a, 0x3, 0x2, 
    0x2, 0x2, 0x199, 0x148, 0x3, 0x2, 0x2, 0x2, 0x199, 0x17b, 0x3, 0x2, 
    0x2, 0x2, 0x199, 0x194, 0x3, 0x2, 0x2, 0x2, 0x19a, 0x19, 0x3, 0x2, 0x2, 
    0x2, 0x19b, 0x19c, 0x7, 0x2a, 0x2, 0x2, 0x19c, 0x1b, 0x3, 0x2, 0x2, 
    0x2, 0x19d, 0x1a0, 0x5, 0x1e, 0x10, 0x2, 0x19e, 0x1a0, 0x5, 0x50, 0x29, 
    0x2, 0x19f, 0x19d, 0x3, 0x2, 0x2, 0x2, 0x19f, 0x19e, 0x3, 0x2, 0x2, 
    0x2, 0x1a0, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x1a1, 0x1a6, 0x5, 0x20, 0x11, 
    0x2, 0x1a2, 0x1a3, 0x7, 0x3a, 0x2, 0x2, 0x1a3, 0x1a5, 0x5, 0x20, 0x11, 
    0x2, 0x1a4, 0x1a2, 0x3, 0x2, 0x2, 0x2, 0x1a5, 0x1a8, 0x3, 0x2, 0x2, 
    0x2, 0x1a6, 0x1a4, 0x3, 0x2, 0x2, 0x2, 0x1a6, 0x1a7, 0x3, 0x2, 0x2, 
    0x2, 0x1a7, 0x1aa, 0x3, 0x2, 0x2, 0x2, 0x1a8, 0x1a6, 0x3, 0x2, 0x2, 
    0x2, 0x1a9, 0x1ab, 0x7, 0x3a, 0x2, 0x2, 0x1aa, 0x1a9, 0x3, 0x2, 0x2, 
    0x2, 0x1aa, 0x1ab, 0x3, 0x2, 0x2, 0x2, 0x1ab, 0x1ac, 0x3, 0x2, 0x2, 
    0x2, 0x1ac, 0x1ad, 0x7, 0x29, 0x2, 0x2, 0x1ad, 0x1f, 0x3, 0x2, 0x2, 
    0x2, 0x1ae, 0x1b7, 0x5, 0x22, 0x12, 0x2, 0x1af, 0x1b7, 0x5, 0x2a, 0x16, 
    0x2, 0x1b0, 0x1b7, 0x5, 0x2c, 0x17, 0x2, 0x1b1, 0x1b7, 0x5, 0x2e, 0x18, 
    0x2, 0x1b2, 0x1b7, 0x5, 0x3a, 0x1e, 0x2, 0x1b3, 0x1b7, 0x5, 0x4a, 0x26, 
    0x2, 0x1b4, 0x1b7, 0x5, 0x4c, 0x27, 0x2, 0x1b5, 0x1b7, 0x5, 0x4e, 0x28, 
    0x2, 0x1b6, 0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1af, 0x3, 0x2, 0x2, 
    0x2, 0x1b6, 0x1b0, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b1, 0x3, 0x2, 0x2, 
    0x2, 0x1b6, 0x1b2, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b3, 0x3, 0x2, 0x2, 
    0x2, 0x1b6, 0x1b4, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b5, 0x3, 0x2, 0x2, 
    0x2, 0x1b7, 0x21, 0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1c9, 0x5, 0x26, 0x14, 
    0x2, 0x1b9, 0x1ca, 0x5, 0x24, 0x13, 0x2, 0x1ba, 0x1bd, 0x5, 0x28, 0x15, 
    0x2, 0x1bb, 0x1be, 0x5, 0xaa, 0x56, 0x2, 0x1bc, 0x1be, 0x5, 0x98, 0x4d, 
    0x2, 0x1bd, 0x1bb, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1bc, 0x3, 0x2, 0x2, 
    0x2, 0x1be, 0x1ca, 0x3, 0x2, 0x2, 0x2, 0x1bf, 0x1c2, 0x7, 0x3c, 0x2, 
    0x2, 0x1c0, 0x1c3, 0x5, 0xaa, 0x56, 0x2, 0x1c1, 0x1c3, 0x5, 0x26, 0x14, 
    0x2, 0x1c2, 0x1c0, 0x3, 0x2, 0x2, 0x2, 0x1c2, 0x1c1, 0x3, 0x2, 0x2, 
    0x2, 0x1c3, 0x1c5, 0x3, 0x2, 0x2, 0x2, 0x1c4, 0x1bf, 0x3, 0x2, 0x2, 
    0x2, 0x1c5, 0x1c8, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c4, 0x3, 0x2, 0x2, 
    0x2, 0x1c6, 0x1c7, 0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1ca, 0x3, 0x2, 0x2, 
    0x2, 0x1c8, 0x1c6, 0x3, 0x2, 0x2, 0x2, 0x1c9, 0x1b9, 0x3, 0x2, 0x2, 
    0x2, 0x1c9, 0x1ba, 0x3, 0x2, 0x2, 0x2, 0x1c9, 0x1c6, 0x3, 0x2, 0x2, 
    0x2, 0x1ca, 0x23, 0x3, 0x2, 0x2, 0x2, 0x1cb, 0x1cc, 0x7, 0x39, 0x2, 
    0x2, 0x1cc, 0x1cf, 0x5, 0x64, 0x33, 0x2, 0x1cd, 0x1ce, 0x7, 0x3c, 0x2, 
    0x2, 0x1ce, 0x1d0, 0x5, 0x64, 0x33, 0x2, 0x1cf, 0x1cd, 0x3, 0x2, 0x2, 
    0x2, 0x1cf, 0x1d0, 0x3, 0x2, 0x2, 0x2, 0x1d0, 0x25, 0x3, 0x2, 0x2, 0x2, 
    0x1d1, 0x1d4, 0x5, 0x64, 0x33, 0x2, 0x1d2, 0x1d4, 0x5, 0x76, 0x3c, 0x2, 
    0x1d3, 0x1d1, 0x3, 0x2, 0x2, 0x2, 0x1d3, 0x1d2, 0x3, 0x2, 0x2, 0x2, 
    0x1d4, 0x1dc, 0x3, 0x2, 0x2, 0x2, 0x1d5, 0x1d8, 0x7, 0x38, 0x2, 0x2, 
    0x1d6, 0x1d9, 0x5, 0x64, 0x33, 0x2, 0x1d7, 0x1d9, 0x5, 0x76, 0x3c, 0x2, 
    0x1d8, 0x1d6, 0x3, 0x2, 0x2, 0x2, 0x1d8, 0x1d7, 0x3, 0x2, 0x2, 0x2, 
    0x1d9, 0x1db, 0x3, 0x2, 0x2, 0x2, 0x1da, 0x1d5, 0x3, 0x2, 0x2, 0x2, 
    0x1db, 0x1de, 0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1da, 0x3, 0x2, 0x2, 0x2, 
    0x1dc, 0x1dd, 0x3, 0x2, 0x2, 0x2, 0x1dd, 0x1e0, 0x3, 0x2, 0x2, 0x2, 
    0x1de, 0x1dc, 0x3, 0x2, 0x2, 0x2, 0x1df, 0x1e1, 0x7, 0x38, 0x2, 0x2, 
    0x1e0, 0x1df, 0x3, 0x2, 0x2, 0x2, 0x1e0, 0x1e1, 0x3, 0x2, 0x2, 0x2, 
    0x1e1, 0x27, 0x3, 0x2, 0x2, 0x2, 0x1e2, 0x1e3, 0x9, 0x2, 0x2, 0x2, 0x1e3, 
    0x29, 0x3, 0x2, 0x2, 0x2, 0x1e4, 0x1e5, 0x7, 0x23, 0x2, 0x2, 0x1e5, 
    0x1e6, 0x5, 0x96, 0x4c, 0x2, 0x1e6, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x1e7, 
    0x1e8, 0x7, 0x24, 0x2, 0x2, 0x1e8, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x1e9, 
    0x1ef, 0x5, 0x30, 0x19, 0x2, 0x1ea, 0x1ef, 0x5, 0x32, 0x1a, 0x2, 0x1eb, 
    0x1ef, 0x5, 0x34, 0x1b, 0x2, 0x1ec, 0x1ef, 0x5, 0x38, 0x1d, 0x2, 0x1ed, 
    0x1ef, 0x5, 0x36, 0x1c, 0x2, 0x1ee, 0x1e9, 0x3, 0x2, 0x2, 0x2, 0x1ee, 
    0x1ea, 0x3, 0x2, 0x2, 0x2, 0x1ee, 0x1eb, 0x3, 0x2, 0x2, 0x2, 0x1ee, 
    0x1ec, 0x3, 0x2, 0x2, 0x2, 0x1ee, 0x1ed, 0x3, 0x2, 0x2, 0x2, 0x1ef, 
    0x2f, 0x3, 0x2, 0x2, 0x2, 0x1f0, 0x1f1, 0x7, 0x26, 0x2, 0x2, 0x1f1, 
    0x31, 0x3, 0x2, 0x2, 0x2, 0x1f2, 0x1f3, 0x7, 0x25, 0x2, 0x2, 0x1f3, 
    0x33, 0x3, 0x2, 0x2, 0x2, 0x1f4, 0x1f6, 0x7, 0x7, 0x2, 0x2, 0x1f5, 0x1f7, 
    0x5, 0x98, 0x4d, 0x2, 0x1f6, 0x1f5, 0x3, 0x2, 0x2, 0x2, 0x1f6, 0x1f7, 
    0x3, 0x2, 0x2, 0x2, 0x1f7, 0x35, 0x3, 0x2, 0x2, 0x2, 0x1f8, 0x1f9, 0x5, 
    0xaa, 0x56, 0x2, 0x1f9, 0x37, 0x3, 0x2, 0x2, 0x2, 0x1fa, 0x200, 0x7, 
    0x8, 0x2, 0x2, 0x1fb, 0x1fe, 0x5, 0x64, 0x33, 0x2, 0x1fc, 0x1fd, 0x7, 
    0x9, 0x2, 0x2, 0x1fd, 0x1ff, 0x5, 0x64, 0x33, 0x2, 0x1fe, 0x1fc, 0x3, 
    0x2, 0x2, 0x2, 0x1fe, 0x1ff, 0x3, 0x2, 0x2, 0x2, 0x1ff, 0x201, 0x3, 
    0x2, 0x2, 0x2, 0x200, 0x1fb, 0x3, 0x2, 0x2, 0x2, 0x200, 0x201, 0x3, 
    0x2, 0x2, 0x2, 0x201, 0x39, 0x3, 0x2, 0x2, 0x2, 0x202, 0x205, 0x5, 0x3c, 
    0x1f, 0x2, 0x203, 0x205, 0x5, 0x3e, 0x20, 0x2, 0x204, 0x202, 0x3, 0x2, 
    0x2, 0x2, 0x204, 0x203, 0x3, 0x2, 0x2, 0x2, 0x205, 0x3b, 0x3, 0x2, 0x2, 
    0x2, 0x206, 0x207, 0x7, 0xa, 0x2, 0x2, 0x207, 0x208, 0x5, 0x46, 0x24, 
    0x2, 0x208, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x209, 0x216, 0x7, 0x9, 0x2, 0x2, 
    0x20a, 0x20c, 0x9, 0x3, 0x2, 0x2, 0x20b, 0x20a, 0x3, 0x2, 0x2, 0x2, 
    0x20c, 0x20f, 0x3, 0x2, 0x2, 0x2, 0x20d, 0x20b, 0x3, 0x2, 0x2, 0x2, 
    0x20d, 0x20e, 0x3, 0x2, 0x2, 0x2, 0x20e, 0x210, 0x3, 0x2, 0x2, 0x2, 
    0x20f, 0x20d, 0x3, 0x2, 0x2, 0x2, 0x210, 0x217, 0x5, 0x48, 0x25, 0x2, 
    0x211, 0x213, 0x9, 0x3, 0x2, 0x2, 0x212, 0x211, 0x3, 0x2, 0x2, 0x2, 
    0x213, 0x214, 0x3, 0x2, 0x2, 0x2, 0x214, 0x212, 0x3, 0x2, 0x2, 0x2, 
    0x214, 0x215, 0x3, 0x2, 0x2, 0x2, 0x215, 0x217, 0x3, 0x2, 0x2, 0x2, 
    0x216, 0x20d, 0x3, 0x2, 0x2, 0x2, 0x216, 0x212, 0x3, 0x2, 0x2, 0x2, 
    0x217, 0x218, 0x3, 0x2, 0x2, 0x2, 0x218, 0x21f, 0x7, 0xa, 0x2, 0x2, 
    0x219, 0x220, 0x7, 0x35, 0x2, 0x2, 0x21a, 0x21b, 0x7, 0x36, 0x2, 0x2, 
    0x21b, 0x21c, 0x5, 0x44, 0x23, 0x2, 0x21c, 0x21d, 0x7, 0x37, 0x2, 0x2, 
    0x21d, 0x220, 0x3, 0x2, 0x2, 0x2, 0x21e, 0x220, 0x5, 0x44, 0x23, 0x2, 
    0x21f, 0x219, 0x3, 0x2, 0x2, 0x2, 0x21f, 0x21a, 0x3, 0x2, 0x2, 0x2, 
    0x21f, 0x21e, 0x3, 0x2, 0x2, 0x2, 0x220, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x221, 
    0x224, 0x7, 0x2a, 0x2, 0x2, 0x222, 0x223, 0x7, 0xb, 0x2, 0x2, 0x223, 
    0x225, 0x7, 0x2a, 0x2, 0x2, 0x224, 0x222, 0x3, 0x2, 0x2, 0x2, 0x224, 
    0x225, 0x3, 0x2, 0x2, 0x2, 0x225, 0x41, 0x3, 0x2, 0x2, 0x2, 0x226, 0x229, 
    0x5, 0x48, 0x25, 0x2, 0x227, 0x228, 0x7, 0xb, 0x2, 0x2, 0x228, 0x22a, 
    0x7, 0x2a, 0x2, 0x2, 0x229, 0x227, 0x3, 0x2, 0x2, 0x2, 0x229, 0x22a, 
    0x3, 0x2, 0x2, 0x2, 0x22a, 0x43, 0x3, 0x2, 0x2, 0x2, 0x22b, 0x230, 0x5, 
    0x40, 0x21, 0x2, 0x22c, 0x22d, 0x7, 0x38, 0x2, 0x2, 0x22d, 0x22f, 0x5, 
    0x40, 0x21, 0x2, 0x22e, 0x22c, 0x3, 0x2, 0x2, 0x2, 0x22f, 0x232, 0x3, 
    0x2, 0x2, 0x2, 0x230, 0x22e, 0x3, 0x2, 0x2, 0x2, 0x230, 0x231, 0x3, 
    0x2, 0x2, 0x2, 0x231, 0x234, 0x3, 0x2, 0x2, 0x2, 0x232, 0x230, 0x3, 
    0x2, 0x2, 0x2, 0x233, 0x235, 0x7, 0x38, 0x2, 0x2, 0x234, 0x233, 0x3, 
    0x2, 0x2, 0x2, 0x234, 0x235, 0x3, 0x2, 0x2, 0x2, 0x235, 0x45, 0x3, 0x2, 
    0x2, 0x2, 0x236, 0x23b, 0x5, 0x42, 0x22, 0x2, 0x237, 0x238, 0x7, 0x38, 
    0x2, 0x2, 0x238, 0x23a, 0x5, 0x42, 0x22, 0x2, 0x239, 0x237, 0x3, 0x2, 
    0x2, 0x2, 0x23a, 0x23d, 0x3, 0x2, 0x2, 0x2, 0x23b, 0x239, 0x3, 0x2, 
    0x2, 0x2, 0x23b, 0x23c, 0x3, 0x2, 0x2, 0x2, 0x23c, 0x47, 0x3, 0x2, 0x2, 
    0x2, 0x23d, 0x23b, 0x3, 0x2, 0x2, 0x2, 0x23e, 0x243, 0x7, 0x2a, 0x2, 
    0x2, 0x23f, 0x240, 0x7, 0x33, 0x2, 0x2, 0x240, 0x242, 0x7, 0x2a, 0x2, 
    0x2, 0x241, 0x23f, 0x3, 0x2, 0x2, 0x2, 0x242, 0x245, 0x3, 0x2, 0x2, 
    0x2, 0x243, 0x241, 0x3, 0x2, 0x2, 0x2, 0x243, 0x244, 0x3, 0x2, 0x2, 
    0x2, 0x244, 0x49, 0x3, 0x2, 0x2, 0x2, 0x245, 0x243, 0x3, 0x2, 0x2, 0x2, 
    0x246, 0x247, 0x7, 0xc, 0x2, 0x2, 0x247, 0x24c, 0x7, 0x2a, 0x2, 0x2, 
    0x248, 0x249, 0x7, 0x38, 0x2, 0x2, 0x249, 0x24b, 0x7, 0x2a, 0x2, 0x2, 
    0x24a, 0x248, 0x3, 0x2, 0x2, 0x2, 0x24b, 0x24e, 0x3, 0x2, 0x2, 0x2, 
    0x24c, 0x24a, 0x3, 0x2, 0x2, 0x2, 0x24c, 0x24d, 0x3, 0x2, 0x2, 0x2, 
    0x24d, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x24e, 0x24c, 0x3, 0x2, 0x2, 0x2, 0x24f, 
    0x250, 0x7, 0xd, 0x2, 0x2, 0x250, 0x255, 0x7, 0x2a, 0x2, 0x2, 0x251, 
    0x252, 0x7, 0x38, 0x2, 0x2, 0x252, 0x254, 0x7, 0x2a, 0x2, 0x2, 0x253, 
    0x251, 0x3, 0x2, 0x2, 0x2, 0x254, 0x257, 0x3, 0x2, 0x2, 0x2, 0x255, 
    0x253, 0x3, 0x2, 0x2, 0x2, 0x255, 0x256, 0x3, 0x2, 0x2, 0x2, 0x256, 
    0x4d, 0x3, 0x2, 0x2, 0x2, 0x257, 0x255, 0x3, 0x2, 0x2, 0x2, 0x258, 0x259, 
    0x7, 0xe, 0x2, 0x2, 0x259, 0x25c, 0x5, 0x64, 0x33, 0x2, 0x25a, 0x25b, 
    0x7, 0x38, 0x2, 0x2, 0x25b, 0x25d, 0x5, 0x64, 0x33, 0x2, 0x25c, 0x25a, 
    0x3, 0x2, 0x2, 0x2, 0x25c, 0x25d, 0x3, 0x2, 0x2, 0x2, 0x25d, 0x4f, 0x3, 
    0x2, 0x2, 0x2, 0x25e, 0x268, 0x5, 0x54, 0x2b, 0x2, 0x25f, 0x268, 0x5, 
    0x56, 0x2c, 0x2, 0x260, 0x268, 0x5, 0x58, 0x2d, 0x2, 0x261, 0x268, 0x5, 
    0x5a, 0x2e, 0x2, 0x262, 0x268, 0x5, 0x5c, 0x2f, 0x2, 0x263, 0x268, 0x5, 
    0x10, 0x9, 0x2, 0x264, 0x268, 0x5, 0x9c, 0x4f, 0x2, 0x265, 0x268, 0x5, 
    0xc, 0x7, 0x2, 0x266, 0x268, 0x5, 0x52, 0x2a, 0x2, 0x267, 0x25e, 0x3, 
    0x2, 0x2, 0x2, 0x267, 0x25f, 0x3, 0x2, 0x2, 0x2, 0x267, 0x260, 0x3, 
    0x2, 0x2, 0x2, 0x267, 0x261, 0x3, 0x2, 0x2, 0x2, 0x267, 0x262, 0x3, 
    0x2, 0x2, 0x2, 0x267, 0x263, 0x3, 0x2, 0x2, 0x2, 0x267, 0x264, 0x3, 
    0x2, 0x2, 0x2, 0x267, 0x265, 0x3, 0x2, 0x2, 0x2, 0x267, 0x266, 0x3, 
    0x2, 0x2, 0x2, 0x268, 0x51, 0x3, 0x2, 0x2, 0x2, 0x269, 0x26d, 0x7, 0x27, 
    0x2, 0x2, 0x26a, 0x26e, 0x5, 0x10, 0x9, 0x2, 0x26b, 0x26e, 0x5, 0x5c, 
    0x2f, 0x2, 0x26c, 0x26e, 0x5, 0x58, 0x2d, 0x2, 0x26d, 0x26a, 0x3, 0x2, 
    0x2, 0x2, 0x26d, 0x26b, 0x3, 0x2, 0x2, 0x2, 0x26d, 0x26c, 0x3, 0x2, 
    0x2, 0x2, 0x26e, 0x53, 0x3, 0x2, 0x2, 0x2, 0x26f, 0x270, 0x7, 0xf, 0x2, 
    0x2, 0x270, 0x271, 0x5, 0x64, 0x33, 0x2, 0x271, 0x272, 0x7, 0x39, 0x2, 
    0x2, 0x272, 0x27a, 0x5, 0x62, 0x32, 0x2, 0x273, 0x274, 0x7, 0x10, 0x2, 
    0x2, 0x274, 0x275, 0x5, 0x64, 0x33, 0x2, 0x275, 0x276, 0x7, 0x39, 0x2, 
    0x2, 0x276, 0x277, 0x5, 0x62, 0x32, 0x2, 0x277, 0x279, 0x3, 0x2, 0x2, 
    0x2, 0x278, 0x273, 0x3, 0x2, 0x2, 0x2, 0x279, 0x27c, 0x3, 0x2, 0x2, 
    0x2, 0x27a, 0x278, 0x3, 0x2, 0x2, 0x2, 0x27a, 0x27b, 0x3, 0x2, 0x2, 
    0x2, 0x27b, 0x280, 0x3, 0x2, 0x2, 0x2, 0x27c, 0x27a, 0x3, 0x2, 0x2, 
    0x2, 0x27d, 0x27e, 0x7, 0x11, 0x2, 0x2, 0x27e, 0x27f, 0x7, 0x39, 0x2, 
    0x2, 0x27f, 0x281, 0x5, 0x62, 0x32, 0x2, 0x280, 0x27d, 0x3, 0x2, 0x2, 
    0x2, 0x280, 0x281, 0x3, 0x2, 0x2, 0x2, 0x281, 0x55, 0x3, 0x2, 0x2, 0x2, 
    0x282, 0x283, 0x7, 0x12, 0x2, 0x2, 0x283, 0x284, 0x5, 0x64, 0x33, 0x2, 
    0x284, 0x285, 0x7, 0x39, 0x2, 0x2, 0x285, 0x289, 0x5, 0x62, 0x32, 0x2, 
    0x286, 0x287, 0x7, 0x11, 0x2, 0x2, 0x287, 0x288, 0x7, 0x39, 0x2, 0x2, 
    0x288, 0x28a, 0x5, 0x62, 0x32, 0x2, 0x289, 0x286, 0x3, 0x2, 0x2, 0x2, 
    0x289, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x28a, 0x57, 0x3, 0x2, 0x2, 0x2, 0x28b, 
    0x28c, 0x7, 0x13, 0x2, 0x2, 0x28c, 0x28d, 0x5, 0x96, 0x4c, 0x2, 0x28d, 
    0x28e, 0x7, 0x14, 0x2, 0x2, 0x28e, 0x28f, 0x5, 0x98, 0x4d, 0x2, 0x28f, 
    0x290, 0x7, 0x39, 0x2, 0x2, 0x290, 0x294, 0x5, 0x62, 0x32, 0x2, 0x291, 
    0x292, 0x7, 0x11, 0x2, 0x2, 0x292, 0x293, 0x7, 0x39, 0x2, 0x2, 0x293, 
    0x295, 0x5, 0x62, 0x32, 0x2, 0x294, 0x291, 0x3, 0x2, 0x2, 0x2, 0x294, 
    0x295, 0x3, 0x2, 0x2, 0x2, 0x295, 0x59, 0x3, 0x2, 0x2, 0x2, 0x296, 0x297, 
    0x7, 0x15, 0x2, 0x2, 0x297, 0x298, 0x7, 0x39, 0x2, 0x2, 0x298, 0x2ae, 
    0x5, 0x62, 0x32, 0x2, 0x299, 0x29a, 0x5, 0x60, 0x31, 0x2, 0x29a, 0x29b, 
    0x7, 0x39, 0x2, 0x2, 0x29b, 0x29c, 0x5, 0x62, 0x32, 0x2, 0x29c, 0x29e, 
    0x3, 0x2, 0x2, 0x2, 0x29d, 0x299, 0x3, 0x2, 0x2, 0x2, 0x29e, 0x29f, 
    0x3, 0x2, 0x2, 0x2, 0x29f, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x29f, 0x2a0, 
    0x3, 0x2, 0x2, 0x2, 0x2a0, 0x2a4, 0x3, 0x2, 0x2, 0x2, 0x2a1, 0x2a2, 
    0x7, 0x11, 0x2, 0x2, 0x2a2, 0x2a3, 0x7, 0x39, 0x2, 0x2, 0x2a3, 0x2a5, 
    0x5, 0x62, 0x32, 0x2, 0x2a4, 0x2a1, 0x3, 0x2, 0x2, 0x2, 0x2a4, 0x2a5, 
    0x3, 0x2, 0x2, 0x2, 0x2a5, 0x2a9, 0x3, 0x2, 0x2, 0x2, 0x2a6, 0x2a7, 
    0x7, 0x16, 0x2, 0x2, 0x2a7, 0x2a8, 0x7, 0x39, 0x2, 0x2, 0x2a8, 0x2aa, 
    0x5, 0x62, 0x32, 0x2, 0x2a9, 0x2a6, 0x3, 0x2, 0x2, 0x2, 0x2a9, 0x2aa, 
    0x3, 0x2, 0x2, 0x2, 0x2aa, 0x2af, 0x3, 0x2, 0x2, 0x2, 0x2ab, 0x2ac, 
    0x7, 0x16, 0x2, 0x2, 0x2ac, 0x2ad, 0x7, 0x39, 0x2, 0x2, 0x2ad, 0x2af, 
    0x5, 0x62, 0x32, 0x2, 0x2ae, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x2ae, 0x2ab, 
    0x3, 0x2, 0x2, 0x2, 0x2af, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x2b0, 0x2b1, 0x7, 
    0x17, 0x2, 0x2, 0x2b1, 0x2b6, 0x5, 0x5e, 0x30, 0x2, 0x2b2, 0x2b3, 0x7, 
    0x38, 0x2, 0x2, 0x2b3, 0x2b5, 0x5, 0x5e, 0x30, 0x2, 0x2b4, 0x2b2, 0x3, 
    0x2, 0x2, 0x2, 0x2b5, 0x2b8, 0x3, 0x2, 0x2, 0x2, 0x2b6, 0x2b4, 0x3, 
    0x2, 0x2, 0x2, 0x2b6, 0x2b7, 0x3, 0x2, 0x2, 0x2, 0x2b7, 0x2b9, 0x3, 
    0x2, 0x2, 0x2, 0x2b8, 0x2b6, 0x3, 0x2, 0x2, 0x2, 0x2b9, 0x2ba, 0x7, 
    0x39, 0x2, 0x2, 0x2ba, 0x2bb, 0x5, 0x62, 0x32, 0x2, 0x2bb, 0x5d, 0x3, 
    0x2, 0x2, 0x2, 0x2bc, 0x2bf, 0x5, 0x64, 0x33, 0x2, 0x2bd, 0x2be, 0x7, 
    0xb, 0x2, 0x2, 0x2be, 0x2c0, 0x5, 0x78, 0x3d, 0x2, 0x2bf, 0x2bd, 0x3, 
    0x2, 0x2, 0x2, 0x2bf, 0x2c0, 0x3, 0x2, 0x2, 0x2, 0x2c0, 0x5f, 0x3, 0x2, 
    0x2, 0x2, 0x2c1, 0x2c7, 0x7, 0x18, 0x2, 0x2, 0x2c2, 0x2c5, 0x5, 0x64, 
    0x33, 0x2, 0x2c3, 0x2c4, 0x7, 0xb, 0x2, 0x2, 0x2c4, 0x2c6, 0x7, 0x2a, 
    0x2, 0x2, 0x2c5, 0x2c3, 0x3, 0x2, 0x2, 0x2, 0x2c5, 0x2c6, 0x3, 0x2, 
    0x2, 0x2, 0x2c6, 0x2c8, 0x3, 0x2, 0x2, 0x2, 0x2c7, 0x2c2, 0x3, 0x2, 
    0x2, 0x2, 0x2c7, 0x2c8, 0x3, 0x2, 0x2, 0x2, 0x2c8, 0x61, 0x3, 0x2, 0x2, 
    0x2, 0x2c9, 0x2d4, 0x5, 0x1e, 0x10, 0x2, 0x2ca, 0x2cb, 0x7, 0x29, 0x2, 
    0x2, 0x2cb, 0x2cd, 0x7, 0x64, 0x2, 0x2, 0x2cc, 0x2ce, 0x5, 0x1c, 0xf, 
    0x2, 0x2cd, 0x2cc, 0x3, 0x2, 0x2, 0x2, 0x2ce, 0x2cf, 0x3, 0x2, 0x2, 
    0x2, 0x2cf, 0x2cd, 0x3, 0x2, 0x2, 0x2, 0x2cf, 0x2d0, 0x3, 0x2, 0x2, 
    0x2, 0x2d0, 0x2d1, 0x3, 0x2, 0x2, 0x2, 0x2d1, 0x2d2, 0x7, 0x65, 0x2, 
    0x2, 0x2d2, 0x2d4, 0x3, 0x2, 0x2, 0x2, 0x2d3, 0x2c9, 0x3, 0x2, 0x2, 
    0x2, 0x2d3, 0x2ca, 0x3, 0x2, 0x2, 0x2, 0x2d4, 0x63, 0x3, 0x2, 0x2, 0x2, 
    0x2d5, 0x2db, 0x5, 0x6c, 0x37, 0x2, 0x2d6, 0x2d7, 0x7, 0xf, 0x2, 0x2, 
    0x2d7, 0x2d8, 0x5, 0x6c, 0x37, 0x2, 0x2d8, 0x2d9, 0x7, 0x11, 0x2, 0x2, 
    0x2d9, 0x2da, 0x5, 0x64, 0x33, 0x2, 0x2da, 0x2dc, 0x3, 0x2, 0x2, 0x2, 
    0x2db, 0x2d6, 0x3, 0x2, 0x2, 0x2, 0x2db, 0x2dc, 0x3, 0x2, 0x2, 0x2, 
    0x2dc, 0x2df, 0x3, 0x2, 0x2, 0x2, 0x2dd, 0x2df, 0x5, 0x68, 0x35, 0x2, 
    0x2de, 0x2d5, 0x3, 0x2, 0x2, 0x2, 0x2de, 0x2dd, 0x3, 0x2, 0x2, 0x2, 
    0x2df, 0x65, 0x3, 0x2, 0x2, 0x2, 0x2e0, 0x2e3, 0x5, 0x6c, 0x37, 0x2, 
    0x2e1, 0x2e3, 0x5, 0x6a, 0x36, 0x2, 0x2e2, 0x2e0, 0x3, 0x2, 0x2, 0x2, 
    0x2e2, 0x2e1, 0x3, 0x2, 0x2, 0x2, 0x2e3, 0x67, 0x3, 0x2, 0x2, 0x2, 0x2e4, 
    0x2e6, 0x7, 0x19, 0x2, 0x2, 0x2e5, 0x2e7, 0x5, 0x18, 0xd, 0x2, 0x2e6, 
    0x2e5, 0x3, 0x2, 0x2, 0x2, 0x2e6, 0x2e7, 0x3, 0x2, 0x2, 0x2, 0x2e7, 
    0x2e8, 0x3, 0x2, 0x2, 0x2, 0x2e8, 0x2e9, 0x7, 0x39, 0x2, 0x2, 0x2e9, 
    0x2ea, 0x5, 0x64, 0x33, 0x2, 0x2ea, 0x69, 0x3, 0x2, 0x2, 0x2, 0x2eb, 
    0x2ed, 0x7, 0x19, 0x2, 0x2, 0x2ec, 0x2ee, 0x5, 0x18, 0xd, 0x2, 0x2ed, 
    0x2ec, 0x3, 0x2, 0x2, 0x2, 0x2ed, 0x2ee, 0x3, 0x2, 0x2, 0x2, 0x2ee, 
    0x2ef, 0x3, 0x2, 0x2, 0x2, 0x2ef, 0x2f0, 0x7, 0x39, 0x2, 0x2, 0x2f0, 
    0x2f1, 0x5, 0x66, 0x34, 0x2, 0x2f1, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x2f2, 
    0x2f7, 0x5, 0x6e, 0x38, 0x2, 0x2f3, 0x2f4, 0x7, 0x1a, 0x2, 0x2, 0x2f4, 
    0x2f6, 0x5, 0x6e, 0x38, 0x2, 0x2f5, 0x2f3, 0x3, 0x2, 0x2, 0x2, 0x2f6, 
    0x2f9, 0x3, 0x2, 0x2, 0x2, 0x2f7, 0x2f5, 0x3, 0x2, 0x2, 0x2, 0x2f7, 
    0x2f8, 0x3, 0x2, 0x2, 0x2, 0x2f8, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x2f9, 0x2f7, 
    0x3, 0x2, 0x2, 0x2, 0x2fa, 0x2ff, 0x5, 0x70, 0x39, 0x2, 0x2fb, 0x2fc, 
    0x7, 0x1b, 0x2, 0x2, 0x2fc, 0x2fe, 0x5, 0x70, 0x39, 0x2, 0x2fd, 0x2fb, 
    0x3, 0x2, 0x2, 0x2, 0x2fe, 0x301, 0x3, 0x2, 0x2, 0x2, 0x2ff, 0x2fd, 
    0x3, 0x2, 0x2, 0x2, 0x2ff, 0x300, 0x3, 0x2, 0x2, 0x2, 0x300, 0x6f, 0x3, 
    0x2, 0x2, 0x2, 0x301, 0x2ff, 0x3, 0x2, 0x2, 0x2, 0x302, 0x303, 0x7, 
    0x1c, 0x2, 0x2, 0x303, 0x306, 0x5, 0x70, 0x39, 0x2, 0x304, 0x306, 0x5, 
    0x72, 0x3a, 0x2, 0x305, 0x302, 0x3, 0x2, 0x2, 0x2, 0x305, 0x304, 0x3, 
    0x2, 0x2, 0x2, 0x306, 0x71, 0x3, 0x2, 0x2, 0x2, 0x307, 0x30d, 0x5, 0x78, 
    0x3d, 0x2, 0x308, 0x309, 0x5, 0x74, 0x3b, 0x2, 0x309, 0x30a, 0x5, 0x78, 
    0x3d, 0x2, 0x30a, 0x30c, 0x3, 0x2, 0x2, 0x2, 0x30b, 0x308, 0x3, 0x2, 
    0x2, 0x2, 0x30c, 0x30f, 0x3, 0x2, 0x2, 0x2, 0x30d, 0x30b, 0x3, 0x2, 
    0x2, 0x2, 0x30d, 0x30e, 0x3, 0x2, 0x2, 0x2, 0x30e, 0x73, 0x3, 0x2, 0x2, 
    0x2, 0x30f, 0x30d, 0x3, 0x2, 0x2, 0x2, 0x310, 0x31e, 0x7, 0x4c, 0x2, 
    0x2, 0x311, 0x31e, 0x7, 0x4d, 0x2, 0x2, 0x312, 0x31e, 0x7, 0x4e, 0x2, 
    0x2, 0x313, 0x31e, 0x7, 0x4f, 0x2, 0x2, 0x314, 0x31e, 0x7, 0x50, 0x2, 
    0x2, 0x315, 0x31e, 0x7, 0x51, 0x2, 0x2, 0x316, 0x31e, 0x7, 0x52, 0x2, 
    0x2, 0x317, 0x31e, 0x7, 0x14, 0x2, 0x2, 0x318, 0x319, 0x7, 0x1c, 0x2, 
    0x2, 0x319, 0x31e, 0x7, 0x14, 0x2, 0x2, 0x31a, 0x31e, 0x7, 0x1d, 0x2, 
    0x2, 0x31b, 0x31c, 0x7, 0x1d, 0x2, 0x2, 0x31c, 0x31e, 0x7, 0x1c, 0x2, 
    0x2, 0x31d, 0x310, 0x3, 0x2, 0x2, 0x2, 0x31d, 0x311, 0x3, 0x2, 0x2, 
    0x2, 0x31d, 0x312, 0x3, 0x2, 0x2, 0x2, 0x31d, 0x313, 0x3, 0x2, 0x2, 
    0x2, 0x31d, 0x314, 0x3, 0x2, 0x2, 0x2, 0x31d, 0x315, 0x3, 0x2, 0x2, 
    0x2, 0x31d, 0x316, 0x3, 0x2, 0x2, 0x2, 0x31d, 0x317, 0x3, 0x2, 0x2, 
    0x2, 0x31d, 0x318, 0x3, 0x2, 0x2, 0x2, 0x31d, 0x31a, 0x3, 0x2, 0x2, 
    0x2, 0x31d, 0x31b, 0x3, 0x2, 0x2, 0x2, 0x31e, 0x75, 0x3, 0x2, 0x2, 0x2, 
    0x31f, 0x320, 0x7, 0x35, 0x2, 0x2, 0x320, 0x321, 0x5, 0x78, 0x3d, 0x2, 
    0x321, 0x77, 0x3, 0x2, 0x2, 0x2, 0x322, 0x327, 0x5, 0x7a, 0x3e, 0x2, 
    0x323, 0x324, 0x7, 0x3f, 0x2, 0x2, 0x324, 0x326, 0x5, 0x7a, 0x3e, 0x2, 
    0x325, 0x323, 0x3, 0x2, 0x2, 0x2, 0x326, 0x329, 0x3, 0x2, 0x2, 0x2, 
    0x327, 0x325, 0x3, 0x2, 0x2, 0x2, 0x327, 0x328, 0x3, 0x2, 0x2, 0x2, 
    0x328, 0x79, 0x3, 0x2, 0x2, 0x2, 0x329, 0x327, 0x3, 0x2, 0x2, 0x2, 0x32a, 
    0x32f, 0x5, 0x7c, 0x3f, 0x2, 0x32b, 0x32c, 0x7, 0x40, 0x2, 0x2, 0x32c, 
    0x32e, 0x5, 0x7c, 0x3f, 0x2, 0x32d, 0x32b, 0x3, 0x2, 0x2, 0x2, 0x32e, 
    0x331, 0x3, 0x2, 0x2, 0x2, 0x32f, 0x32d, 0x3, 0x2, 0x2, 0x2, 0x32f, 
    0x330, 0x3, 0x2, 0x2, 0x2, 0x330, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x331, 0x32f, 
    0x3, 0x2, 0x2, 0x2, 0x332, 0x337, 0x5, 0x7e, 0x40, 0x2, 0x333, 0x334, 
    0x7, 0x41, 0x2, 0x2, 0x334, 0x336, 0x5, 0x7e, 0x40, 0x2, 0x335, 0x333, 
    0x3, 0x2, 0x2, 0x2, 0x336, 0x339, 0x3, 0x2, 0x2, 0x2, 0x337, 0x335, 
    0x3, 0x2, 0x2, 0x2, 0x337, 0x338, 0x3, 0x2, 0x2, 0x2, 0x338, 0x7d, 0x3, 
    0x2, 0x2, 0x2, 0x339, 0x337, 0x3, 0x2, 0x2, 0x2, 0x33a, 0x33f, 0x5, 
    0x80, 0x41, 0x2, 0x33b, 0x33c, 0x9, 0x4, 0x2, 0x2, 0x33c, 0x33e, 0x5, 
    0x80, 0x41, 0x2, 0x33d, 0x33b, 0x3, 0x2, 0x2, 0x2, 0x33e, 0x341, 0x3, 
    0x2, 0x2, 0x2, 0x33f, 0x33d, 0x3, 0x2, 0x2, 0x2, 0x33f, 0x340, 0x3, 
    0x2, 0x2, 0x2, 0x340, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x341, 0x33f, 0x3, 0x2, 
    0x2, 0x2, 0x342, 0x347, 0x5, 0x82, 0x42, 0x2, 0x343, 0x344, 0x9, 0x5, 
    0x2, 0x2, 0x344, 0x346, 0x5, 0x82, 0x42, 0x2, 0x345, 0x343, 0x3, 0x2, 
    0x2, 0x2, 0x346, 0x349, 0x3, 0x2, 0x2, 0x2, 0x347, 0x345, 0x3, 0x2, 
    0x2, 0x2, 0x347, 0x348, 0x3, 0x2, 0x2, 0x2, 0x348, 0x81, 0x3, 0x2, 0x2, 
    0x2, 0x349, 0x347, 0x3, 0x2, 0x2, 0x2, 0x34a, 0x34f, 0x5, 0x84, 0x43, 
    0x2, 0x34b, 0x34c, 0x9, 0x6, 0x2, 0x2, 0x34c, 0x34e, 0x5, 0x84, 0x43, 
    0x2, 0x34d, 0x34b, 0x3, 0x2, 0x2, 0x2, 0x34e, 0x351, 0x3, 0x2, 0x2, 
    0x2, 0x34f, 0x34d, 0x3, 0x2, 0x2, 0x2, 0x34f, 0x350, 0x3, 0x2, 0x2, 
    0x2, 0x350, 0x83, 0x3, 0x2, 0x2, 0x2, 0x351, 0x34f, 0x3, 0x2, 0x2, 0x2, 
    0x352, 0x353, 0x9, 0x7, 0x2, 0x2, 0x353, 0x356, 0x5, 0x84, 0x43, 0x2, 
    0x354, 0x356, 0x5, 0x86, 0x44, 0x2, 0x355, 0x352, 0x3, 0x2, 0x2, 0x2, 
    0x355, 0x354, 0x3, 0x2, 0x2, 0x2, 0x356, 0x85, 0x3, 0x2, 0x2, 0x2, 0x357, 
    0x35a, 0x5, 0x88, 0x45, 0x2, 0x358, 0x359, 0x7, 0x3b, 0x2, 0x2, 0x359, 
    0x35b, 0x5, 0x84, 0x43, 0x2, 0x35a, 0x358, 0x3, 0x2, 0x2, 0x2, 0x35a, 
    0x35b, 0x3, 0x2, 0x2, 0x2, 0x35b, 0x87, 0x3, 0x2, 0x2, 0x2, 0x35c, 0x35e, 
    0x7, 0x28, 0x2, 0x2, 0x35d, 0x35c, 0x3, 0x2, 0x2, 0x2, 0x35d, 0x35e, 
    0x3, 0x2, 0x2, 0x2, 0x35e, 0x35f, 0x3, 0x2, 0x2, 0x2, 0x35f, 0x363, 
    0x5, 0x8a, 0x46, 0x2, 0x360, 0x362, 0x5, 0x8e, 0x48, 0x2, 0x361, 0x360, 
    0x3, 0x2, 0x2, 0x2, 0x362, 0x365, 0x3, 0x2, 0x2, 0x2, 0x363, 0x361, 
    0x3, 0x2, 0x2, 0x2, 0x363, 0x364, 0x3, 0x2, 0x2, 0x2, 0x364, 0x89, 0x3, 
    0x2, 0x2, 0x2, 0x365, 0x363, 0x3, 0x2, 0x2, 0x2, 0x366, 0x369, 0x7, 
    0x36, 0x2, 0x2, 0x367, 0x36a, 0x5, 0xaa, 0x56, 0x2, 0x368, 0x36a, 0x5, 
    0x8c, 0x47, 0x2, 0x369, 0x367, 0x3, 0x2, 0x2, 0x2, 0x369, 0x368, 0x3, 
    0x2, 0x2, 0x2, 0x369, 0x36a, 0x3, 0x2, 0x2, 0x2, 0x36a, 0x36b, 0x3, 
    0x2, 0x2, 0x2, 0x36b, 0x382, 0x7, 0x37, 0x2, 0x2, 0x36c, 0x36e, 0x7, 
    0x3d, 0x2, 0x2, 0x36d, 0x36f, 0x5, 0x8c, 0x47, 0x2, 0x36e, 0x36d, 0x3, 
    0x2, 0x2, 0x2, 0x36e, 0x36f, 0x3, 0x2, 0x2, 0x2, 0x36f, 0x370, 0x3, 
    0x2, 0x2, 0x2, 0x370, 0x382, 0x7, 0x3e, 0x2, 0x2, 0x371, 0x373, 0x7, 
    0x4a, 0x2, 0x2, 0x372, 0x374, 0x5, 0x9a, 0x4e, 0x2, 0x373, 0x372, 0x3, 
    0x2, 0x2, 0x2, 0x373, 0x374, 0x3, 0x2, 0x2, 0x2, 0x374, 0x375, 0x3, 
    0x2, 0x2, 0x2, 0x375, 0x382, 0x7, 0x4b, 0x2, 0x2, 0x376, 0x382, 0x7, 
    0x2a, 0x2, 0x2, 0x377, 0x382, 0x7, 0x4, 0x2, 0x2, 0x378, 0x37a, 0x7, 
    0x3, 0x2, 0x2, 0x379, 0x378, 0x3, 0x2, 0x2, 0x2, 0x37a, 0x37b, 0x3, 
    0x2, 0x2, 0x2, 0x37b, 0x379, 0x3, 0x2, 0x2, 0x2, 0x37b, 0x37c, 0x3, 
    0x2, 0x2, 0x2, 0x37c, 0x382, 0x3, 0x2, 0x2, 0x2, 0x37d, 0x382, 0x7, 
    0x34, 0x2, 0x2, 0x37e, 0x382, 0x7, 0x1e, 0x2, 0x2, 0x37f, 0x382, 0x7, 
    0x1f, 0x2, 0x2, 0x380, 0x382, 0x7, 0x20, 0x2, 0x2, 0x381, 0x366, 0x3, 
    0x2, 0x2, 0x2, 0x381, 0x36c, 0x3, 0x2, 0x2, 0x2, 0x381, 0x371, 0x3, 
    0x2, 0x2, 0x2, 0x381, 0x376, 0x3, 0x2, 0x2, 0x2, 0x381, 0x377, 0x3, 
    0x2, 0x2, 0x2, 0x381, 0x379, 0x3, 0x2, 0x2, 0x2, 0x381, 0x37d, 0x3, 
    0x2, 0x2, 0x2, 0x381, 0x37e, 0x3, 0x2, 0x2, 0x2, 0x381, 0x37f, 0x3, 
    0x2, 0x2, 0x2, 0x381, 0x380, 0x3, 0x2, 0x2, 0x2, 0x382, 0x8b, 0x3, 0x2, 
    0x2, 0x2, 0x383, 0x386, 0x5, 0x64, 0x33, 0x2, 0x384, 0x386, 0x5, 0x76, 
    0x3c, 0x2, 0x385, 0x383, 0x3, 0x2, 0x2, 0x2, 0x385, 0x384, 0x3, 0x2, 
    0x2, 0x2, 0x386, 0x395, 0x3, 0x2, 0x2, 0x2, 0x387, 0x396, 0x5, 0xa4, 
    0x53, 0x2, 0x388, 0x38b, 0x7, 0x38, 0x2, 0x2, 0x389, 0x38c, 0x5, 0x64, 
    0x33, 0x2, 0x38a, 0x38c, 0x5, 0x76, 0x3c, 0x2, 0x38b, 0x389, 0x3, 0x2, 
    0x2, 0x2, 0x38b, 0x38a, 0x3, 0x2, 0x2, 0x2, 0x38c, 0x38e, 0x3, 0x2, 
    0x2, 0x2, 0x38d, 0x388, 0x3, 0x2, 0x2, 0x2, 0x38e, 0x391, 0x3, 0x2, 
    0x2, 0x2, 0x38f, 0x38d, 0x3, 0x2, 0x2, 0x2, 0x38f, 0x390, 0x3, 0x2, 
    0x2, 0x2, 0x390, 0x393, 0x3, 0x2, 0x2, 0x2, 0x391, 0x38f, 0x3, 0x2, 
    0x2, 0x2, 0x392, 0x394, 0x7, 0x38, 0x2, 0x2, 0x393, 0x392, 0x3, 0x2, 
    0x2, 0x2, 0x393, 0x394, 0x3, 0x2, 0x2, 0x2, 0x394, 0x396, 0x3, 0x2, 
    0x2, 0x2, 0x395, 0x387, 0x3, 0x2, 0x2, 0x2, 0x395, 0x38f, 0x3, 0x2, 
    0x2, 0x2, 0x396, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x397, 0x399, 0x7, 0x36, 
    0x2, 0x2, 0x398, 0x39a, 0x5, 0x9e, 0x50, 0x2, 0x399, 0x398, 0x3, 0x2, 
    0x2, 0x2, 0x399, 0x39a, 0x3, 0x2, 0x2, 0x2, 0x39a, 0x39b, 0x3, 0x2, 
    0x2, 0x2, 0x39b, 0x3a3, 0x7, 0x37, 0x2, 0x2, 0x39c, 0x39d, 0x7, 0x3d, 
    0x2, 0x2, 0x39d, 0x39e, 0x5, 0x90, 0x49, 0x2, 0x39e, 0x39f, 0x7, 0x3e, 
    0x2, 0x2, 0x39f, 0x3a3, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x3a1, 0x7, 0x33, 
    0x2, 0x2, 0x3a1, 0x3a3, 0x7, 0x2a, 0x2, 0x2, 0x3a2, 0x397, 0x3, 0x2, 
    0x2, 0x2, 0x3a2, 0x39c, 0x3, 0x2, 0x2, 0x2, 0x3a2, 0x3a0, 0x3, 0x2, 
    0x2, 0x2, 0x3a3, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x3a4, 0x3a9, 0x5, 0x92, 
    0x4a, 0x2, 0x3a5, 0x3a6, 0x7, 0x38, 0x2, 0x2, 0x3a6, 0x3a8, 0x5, 0x92, 
    0x4a, 0x2, 0x3a7, 0x3a5, 0x3, 0x2, 0x2, 0x2, 0x3a8, 0x3ab, 0x3, 0x2, 
    0x2, 0x2, 0x3a9, 0x3a7, 0x3, 0x2, 0x2, 0x2, 0x3a9, 0x3aa, 0x3, 0x2, 
    0x2, 0x2, 0x3aa, 0x3ad, 0x3, 0x2, 0x2, 0x2, 0x3ab, 0x3a9, 0x3, 0x2, 
    0x2, 0x2, 0x3ac, 0x3ae, 0x7, 0x38, 0x2, 0x2, 0x3ad, 0x3ac, 0x3, 0x2, 
    0x2, 0x2, 0x3ad, 0x3ae, 0x3, 0x2, 0x2, 0x2, 0x3ae, 0x91, 0x3, 0x2, 0x2, 
    0x2, 0x3af, 0x3bb, 0x5, 0x64, 0x33, 0x2, 0x3b0, 0x3b2, 0x5, 0x64, 0x33, 
    0x2, 0x3b1, 0x3b0, 0x3, 0x2, 0x2, 0x2, 0x3b1, 0x3b2, 0x3, 0x2, 0x2, 
    0x2, 0x3b2, 0x3b3, 0x3, 0x2, 0x2, 0x2, 0x3b3, 0x3b5, 0x7, 0x39, 0x2, 
    0x2, 0x3b4, 0x3b6, 0x5, 0x64, 0x33, 0x2, 0x3b5, 0x3b4, 0x3, 0x2, 0x2, 
    0x2, 0x3b5, 0x3b6, 0x3, 0x2, 0x2, 0x2, 0x3b6, 0x3b8, 0x3, 0x2, 0x2, 
    0x2, 0x3b7, 0x3b9, 0x5, 0x94, 0x4b, 0x2, 0x3b8, 0x3b7, 0x3, 0x2, 0x2, 
    0x2, 0x3b8, 0x3b9, 0x3, 0x2, 0x2, 0x2, 0x3b9, 0x3bb, 0x3, 0x2, 0x2, 
    0x2, 0x3ba, 0x3af, 0x3, 0x2, 0x2, 0x2, 0x3ba, 0x3b1, 0x3, 0x2, 0x2, 
    0x2, 0x3bb, 0x93, 0x3, 0x2, 0x2, 0x2, 0x3bc, 0x3be, 0x7, 0x39, 0x2, 
    0x2, 0x3bd, 0x3bf, 0x5, 0x64, 0x33, 0x2, 0x3be, 0x3bd, 0x3, 0x2, 0x2, 
    0x2, 0x3be, 0x3bf, 0x3, 0x2, 0x2, 0x2, 0x3bf, 0x95, 0x3, 0x2, 0x2, 0x2, 
    0x3c0, 0x3c3, 0x5, 0x78, 0x3d, 0x2, 0x3c1, 0x3c3, 0x5, 0x76, 0x3c, 0x2, 
    0x3c2, 0x3c0, 0x3, 0x2, 0x2, 0x2, 0x3c2, 0x3c1, 0x3, 0x2, 0x2, 0x2, 
    0x3c3, 0x3cb, 0x3, 0x2, 0x2, 0x2, 0x3c4, 0x3c7, 0x7, 0x38, 0x2, 0x2, 
    0x3c5, 0x3c8, 0x5, 0x78, 0x3d, 0x2, 0x3c6, 0x3c8, 0x5, 0x76, 0x3c, 0x2, 
    0x3c7, 0x3c5, 0x3, 0x2, 0x2, 0x2, 0x3c7, 0x3c6, 0x3, 0x2, 0x2, 0x2, 
    0x3c8, 0x3ca, 0x3, 0x2, 0x2, 0x2, 0x3c9, 0x3c4, 0x3, 0x2, 0x2, 0x2, 
    0x3ca, 0x3cd, 0x3, 0x2, 0x2, 0x2, 0x3cb, 0x3c9, 0x3, 0x2, 0x2, 0x2, 
    0x3cb, 0x3cc, 0x3, 0x2, 0x2, 0x2, 0x3cc, 0x3cf, 0x3, 0x2, 0x2, 0x2, 
    0x3cd, 0x3cb, 0x3, 0x2, 0x2, 0x2, 0x3ce, 0x3d0, 0x7, 0x38, 0x2, 0x2, 
    0x3cf, 0x3ce, 0x3, 0x2, 0x2, 0x2, 0x3cf, 0x3d0, 0x3, 0x2, 0x2, 0x2, 
    0x3d0, 0x97, 0x3, 0x2, 0x2, 0x2, 0x3d1, 0x3d6, 0x5, 0x64, 0x33, 0x2, 
    0x3d2, 0x3d3, 0x7, 0x38, 0x2, 0x2, 0x3d3, 0x3d5, 0x5, 0x64, 0x33, 0x2, 
    0x3d4, 0x3d2, 0x3, 0x2, 0x2, 0x2, 0x3d5, 0x3d8, 0x3, 0x2, 0x2, 0x2, 
    0x3d6, 0x3d4, 0x3, 0x2, 0x2, 0x2, 0x3d6, 0x3d7, 0x3, 0x2, 0x2, 0x2, 
    0x3d7, 0x3da, 0x3, 0x2, 0x2, 0x2, 0x3d8, 0x3d6, 0x3, 0x2, 0x2, 0x2, 
    0x3d9, 0x3db, 0x7, 0x38, 0x2, 0x2, 0x3da, 0x3d9, 0x3, 0x2, 0x2, 0x2, 
    0x3da, 0x3db, 0x3, 0x2, 0x2, 0x2, 0x3db, 0x99, 0x3, 0x2, 0x2, 0x2, 0x3dc, 
    0x3dd, 0x5, 0x64, 0x33, 0x2, 0x3dd, 0x3de, 0x7, 0x39, 0x2, 0x2, 0x3de, 
    0x3df, 0x5, 0x64, 0x33, 0x2, 0x3df, 0x3e3, 0x3, 0x2, 0x2, 0x2, 0x3e0, 
    0x3e1, 0x7, 0x3b, 0x2, 0x2, 0x3e1, 0x3e3, 0x5, 0x78, 0x3d, 0x2, 0x3e2, 
    0x3dc, 0x3, 0x2, 0x2, 0x2, 0x3e2, 0x3e0, 0x3, 0x2, 0x2, 0x2, 0x3e3, 
    0x3f6, 0x3, 0x2, 0x2, 0x2, 0x3e4, 0x3f7, 0x5, 0xa4, 0x53, 0x2, 0x3e5, 
    0x3ec, 0x7, 0x38, 0x2, 0x2, 0x3e6, 0x3e7, 0x5, 0x64, 0x33, 0x2, 0x3e7, 
    0x3e8, 0x7, 0x39, 0x2, 0x2, 0x3e8, 0x3e9, 0x5, 0x64, 0x33, 0x2, 0x3e9, 
    0x3ed, 0x3, 0x2, 0x2, 0x2, 0x3ea, 0x3eb, 0x7, 0x3b, 0x2, 0x2, 0x3eb, 
    0x3ed, 0x5, 0x78, 0x3d, 0x2, 0x3ec, 0x3e6, 0x3, 0x2, 0x2, 0x2, 0x3ec, 
    0x3ea, 0x3, 0x2, 0x2, 0x2, 0x3ed, 0x3ef, 0x3, 0x2, 0x2, 0x2, 0x3ee, 
    0x3e5, 0x3, 0x2, 0x2, 0x2, 0x3ef, 0x3f2, 0x3, 0x2, 0x2, 0x2, 0x3f0, 
    0x3ee, 0x3, 0x2, 0x2, 0x2, 0x3f0, 0x3f1, 0x3, 0x2, 0x2, 0x2, 0x3f1, 
    0x3f4, 0x3, 0x2, 0x2, 0x2, 0x3f2, 0x3f0, 0x3, 0x2, 0x2, 0x2, 0x3f3, 
    0x3f5, 0x7, 0x38, 0x2, 0x2, 0x3f4, 0x3f3, 0x3, 0x2, 0x2, 0x2, 0x3f4, 
    0x3f5, 0x3, 0x2, 0x2, 0x2, 0x3f5, 0x3f7, 0x3, 0x2, 0x2, 0x2, 0x3f6, 
    0x3e4, 0x3, 0x2, 0x2, 0x2, 0x3f6, 0x3f0, 0x3, 0x2, 0x2, 0x2, 0x3f7, 
    0x40d, 0x3, 0x2, 0x2, 0x2, 0x3f8, 0x3fb, 0x5, 0x64, 0x33, 0x2, 0x3f9, 
    0x3fb, 0x5, 0x76, 0x3c, 0x2, 0x3fa, 0x3f8, 0x3, 0x2, 0x2, 0x2, 0x3fa, 
    0x3f9, 0x3, 0x2, 0x2, 0x2, 0x3fb, 0x40a, 0x3, 0x2, 0x2, 0x2, 0x3fc, 
    0x40b, 0x5, 0xa4, 0x53, 0x2, 0x3fd, 0x400, 0x7, 0x38, 0x2, 0x2, 0x3fe, 
    0x401, 0x5, 0x64, 0x33, 0x2, 0x3ff, 0x401, 0x5, 0x76, 0x3c, 0x2, 0x400, 
    0x3fe, 0x3, 0x2, 0x2, 0x2, 0x400, 0x3ff, 0x3, 0x2, 0x2, 0x2, 0x401, 
    0x403, 0x3, 0x2, 0x2, 0x2, 0x402, 0x3fd, 0x3, 0x2, 0x2, 0x2, 0x403, 
    0x406, 0x3, 0x2, 0x2, 0x2, 0x404, 0x402, 0x3, 0x2, 0x2, 0x2, 0x404, 
    0x405, 0x3, 0x2, 0x2, 0x2, 0x405, 0x408, 0x3, 0x2, 0x2, 0x2, 0x406, 
    0x404, 0x3, 0x2, 0x2, 0x2, 0x407, 0x409, 0x7, 0x38, 0x2, 0x2, 0x408, 
    0x407, 0x3, 0x2, 0x2, 0x2, 0x408, 0x409, 0x3, 0x2, 0x2, 0x2, 0x409, 
    0x40b, 0x3, 0x2, 0x2, 0x2, 0x40a, 0x3fc, 0x3, 0x2, 0x2, 0x2, 0x40a, 
    0x404, 0x3, 0x2, 0x2, 0x2, 0x40b, 0x40d, 0x3, 0x2, 0x2, 0x2, 0x40c, 
    0x3e2, 0x3, 0x2, 0x2, 0x2, 0x40c, 0x3fa, 0x3, 0x2, 0x2, 0x2, 0x40d, 
    0x9b, 0x3, 0x2, 0x2, 0x2, 0x40e, 0x40f, 0x7, 0x21, 0x2, 0x2, 0x40f, 
    0x415, 0x7, 0x2a, 0x2, 0x2, 0x410, 0x412, 0x7, 0x36, 0x2, 0x2, 0x411, 
    0x413, 0x5, 0x9e, 0x50, 0x2, 0x412, 0x411, 0x3, 0x2, 0x2, 0x2, 0x412, 
    0x413, 0x3, 0x2, 0x2, 0x2, 0x413, 0x414, 0x3, 0x2, 0x2, 0x2, 0x414, 
    0x416, 0x7, 0x37, 0x2, 0x2, 0x415, 0x410, 0x3, 0x2, 0x2, 0x2, 0x415, 
    0x416, 0x3, 0x2, 0x2, 0x2, 0x416, 0x417, 0x3, 0x2, 0x2, 0x2, 0x417, 
    0x418, 0x7, 0x39, 0x2, 0x2, 0x418, 0x419, 0x5, 0x62, 0x32, 0x2, 0x419, 
    0x9d, 0x3, 0x2, 0x2, 0x2, 0x41a, 0x41f, 0x5, 0xa0, 0x51, 0x2, 0x41b, 
    0x41c, 0x7, 0x38, 0x2, 0x2, 0x41c, 0x41e, 0x5, 0xa0, 0x51, 0x2, 0x41d, 
    0x41b, 0x3, 0x2, 0x2, 0x2, 0x41e, 0x421, 0x3, 0x2, 0x2, 0x2, 0x41f, 
    0x41d, 0x3, 0x2, 0x2, 0x2, 0x41f, 0x420, 0x3, 0x2, 0x2, 0x2, 0x420, 
    0x423, 0x3, 0x2, 0x2, 0x2, 0x421, 0x41f, 0x3, 0x2, 0x2, 0x2, 0x422, 
    0x424, 0x7, 0x38, 0x2, 0x2, 0x423, 0x422, 0x3, 0x2, 0x2, 0x2, 0x423, 
    0x424, 0x3, 0x2, 0x2, 0x2, 0x424, 0x9f, 0x3, 0x2, 0x2, 0x2, 0x425, 0x427, 
    0x5, 0x64, 0x33, 0x2, 0x426, 0x428, 0x5, 0xa4, 0x53, 0x2, 0x427, 0x426, 
    0x3, 0x2, 0x2, 0x2, 0x427, 0x428, 0x3, 0x2, 0x2, 0x2, 0x428, 0x432, 
    0x3, 0x2, 0x2, 0x2, 0x429, 0x42a, 0x5, 0x64, 0x33, 0x2, 0x42a, 0x42b, 
    0x7, 0x3c, 0x2, 0x2, 0x42b, 0x42c, 0x5, 0x64, 0x33, 0x2, 0x42c, 0x432, 
    0x3, 0x2, 0x2, 0x2, 0x42d, 0x42e, 0x7, 0x3b, 0x2, 0x2, 0x42e, 0x432, 
    0x5, 0x64, 0x33, 0x2, 0x42f, 0x430, 0x7, 0x35, 0x2, 0x2, 0x430, 0x432, 
    0x5, 0x64, 0x33, 0x2, 0x431, 0x425, 0x3, 0x2, 0x2, 0x2, 0x431, 0x429, 
    0x3, 0x2, 0x2, 0x2, 0x431, 0x42d, 0x3, 0x2, 0x2, 0x2, 0x431, 0x42f, 
    0x3, 0x2, 0x2, 0x2, 0x432, 0xa1, 0x3, 0x2, 0x2, 0x2, 0x433, 0x436, 0x5, 
    0xa4, 0x53, 0x2, 0x434, 0x436, 0x5, 0xa6, 0x54, 0x2, 0x435, 0x433, 0x3, 
    0x2, 0x2, 0x2, 0x435, 0x434, 0x3, 0x2, 0x2, 0x2, 0x436, 0xa3, 0x3, 0x2, 
    0x2, 0x2, 0x437, 0x439, 0x7, 0x27, 0x2, 0x2, 0x438, 0x437, 0x3, 0x2, 
    0x2, 0x2, 0x438, 0x439, 0x3, 0x2, 0x2, 0x2, 0x439, 0x43a, 0x3, 0x2, 
    0x2, 0x2, 0x43a, 0x43b, 0x7, 0x13, 0x2, 0x2, 0x43b, 0x43c, 0x5, 0x96, 
    0x4c, 0x2, 0x43c, 0x43d, 0x7, 0x14, 0x2, 0x2, 0x43d, 0x43f, 0x5, 0x6c, 
    0x37, 0x2, 0x43e, 0x440, 0x5, 0xa2, 0x52, 0x2, 0x43f, 0x43e, 0x3, 0x2, 
    0x2, 0x2, 0x43f, 0x440, 0x3, 0x2, 0x2, 0x2, 0x440, 0xa5, 0x3, 0x2, 0x2, 
    0x2, 0x441, 0x442, 0x7, 0xf, 0x2, 0x2, 0x442, 0x444, 0x5, 0x66, 0x34, 
    0x2, 0x443, 0x445, 0x5, 0xa2, 0x52, 0x2, 0x444, 0x443, 0x3, 0x2, 0x2, 
    0x2, 0x444, 0x445, 0x3, 0x2, 0x2, 0x2, 0x445, 0xa7, 0x3, 0x2, 0x2, 0x2, 
    0x446, 0x447, 0x7, 0x2a, 0x2, 0x2, 0x447, 0xa9, 0x3, 0x2, 0x2, 0x2, 
    0x448, 0x44a, 0x7, 0x22, 0x2, 0x2, 0x449, 0x44b, 0x5, 0xac, 0x57, 0x2, 
    0x44a, 0x449, 0x3, 0x2, 0x2, 0x2, 0x44a, 0x44b, 0x3, 0x2, 0x2, 0x2, 
    0x44b, 0xab, 0x3, 0x2, 0x2, 0x2, 0x44c, 0x44d, 0x7, 0x9, 0x2, 0x2, 0x44d, 
    0x450, 0x5, 0x64, 0x33, 0x2, 0x44e, 0x450, 0x5, 0x98, 0x4d, 0x2, 0x44f, 
    0x44c, 0x3, 0x2, 0x2, 0x2, 0x44f, 0x44e, 0x3, 0x2, 0x2, 0x2, 0x450, 
    0xad, 0x3, 0x2, 0x2, 0x2, 0xa8, 0xb3, 0xb7, 0xb9, 0xc2, 0xcb, 0xce, 
    0xd5, 0xdb, 0xe5, 0xec, 0xf3, 0xf9, 0xfd, 0x103, 0x109, 0x10d, 0x114, 
    0x116, 0x118, 0x11d, 0x11f, 0x121, 0x125, 0x12b, 0x12f, 0x136, 0x138, 
    0x13a, 0x13f, 0x141, 0x146, 0x14b, 0x151, 0x155, 0x15b, 0x161, 0x165, 
    0x16c, 0x16e, 0x170, 0x175, 0x177, 0x179, 0x17d, 0x183, 0x187, 0x18e, 
    0x190, 0x192, 0x197, 0x199, 0x19f, 0x1a6, 0x1aa, 0x1b6, 0x1bd, 0x1c2, 
    0x1c6, 0x1c9, 0x1cf, 0x1d3, 0x1d8, 0x1dc, 0x1e0, 0x1ee, 0x1f6, 0x1fe, 
    0x200, 0x204, 0x20d, 0x214, 0x216, 0x21f, 0x224, 0x229, 0x230, 0x234, 
    0x23b, 0x243, 0x24c, 0x255, 0x25c, 0x267, 0x26d, 0x27a, 0x280, 0x289, 
    0x294, 0x29f, 0x2a4, 0x2a9, 0x2ae, 0x2b6, 0x2bf, 0x2c5, 0x2c7, 0x2cf, 
    0x2d3, 0x2db, 0x2de, 0x2e2, 0x2e6, 0x2ed, 0x2f7, 0x2ff, 0x305, 0x30d, 
    0x31d, 0x327, 0x32f, 0x337, 0x33f, 0x347, 0x34f, 0x355, 0x35a, 0x35d, 
    0x363, 0x369, 0x36e, 0x373, 0x37b, 0x381, 0x385, 0x38b, 0x38f, 0x393, 
    0x395, 0x399, 0x3a2, 0x3a9, 0x3ad, 0x3b1, 0x3b5, 0x3b8, 0x3ba, 0x3be, 
    0x3c2, 0x3c7, 0x3cb, 0x3cf, 0x3d6, 0x3da, 0x3e2, 0x3ec, 0x3f0, 0x3f4, 
    0x3f6, 0x3fa, 0x400, 0x404, 0x408, 0x40a, 0x40c, 0x412, 0x415, 0x41f, 
    0x423, 0x427, 0x431, 0x435, 0x438, 0x43f, 0x444, 0x44a, 0x44f, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

pyxasmParser::Initializer pyxasmParser::_init;

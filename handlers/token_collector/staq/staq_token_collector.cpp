/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "staq_token_collector.hpp"

#include "InstructionIterator.hpp"
#include "qalloc.hpp"
#include "qrt_mapper.hpp"
#include "xacc.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <Utils.hpp>
#include <iostream>
#include <memory>
#include <set>

#include "clang/Basic/TokenKinds.h"
#include "clang/Parse/Parser.h"

using namespace cppmicroservices;
using namespace clang;

namespace {

/**
 */
class US_ABI_LOCAL StaqTokenCollectorActivator : public BundleActivator {

public:
  StaqTokenCollectorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto st = std::make_shared<qcor::StaqTokenCollector>();
    context.RegisterService<qcor::TokenCollector>(st);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(StaqTokenCollectorActivator)

namespace qcor {

static const std::map<std::string, std::string> gates{
    // "u3", "u2",   "u1", "ccx", cu1, cu3
    {"cx", "CX"},      {"id", "I"},    {"x", "X"},     {"y", "Y"},
    {"z", "Z"},        {"h", "H"},     {"s", "S"},     {"sdg", "Sdg"},
    {"t", "T"},        {"tdg", "Tdg"}, {"rx", "Rx"},   {"ry", "Ry"},
    {"rz", "Rz"},      {"cz", "CZ"},   {"cy", "CY"},   {"swap", "Swap"},
    {"ccx", "CCX"},    {"ch", "CH"},   {"crz", "CRZ"}, {"measure", "Measure"},
    {"reset", "Reset"}};

void StaqTokenCollector::collect(clang::Preprocessor &PP,
                                 clang::CachedTokens &Toks,
                                 std::vector<std::string> bufferNames,
                                 std::stringstream &ss, const std::string &kernel_name) {

  // I need to know of any allocated buffers.
  std::stringstream sss, xx, put_this_after;

  for (auto &b : bufferNames) {
    // Note - we don't know the size of the buffer
    // at this point, so just create one with max size
    // and we can provide an IR Pass later that updates it
    auto q = qalloc(1000);
    q.setNameAndStore(b.c_str());

    xx << "qreg " << b << "[" << q.size() << "];\n";
  }

  // Need to place the above qreg call after the OPENQASM and include calls
  auto open_qasm_iter = Toks.end();
  auto include_iter = Toks.end();
  for (auto iter = Toks.begin(); iter != Toks.end(); ++iter) {
    if (PP.getSpelling(*iter) == "OPENQASM") {
      open_qasm_iter = iter;
    }
    if (PP.getSpelling(*iter) == "include") {
      include_iter = iter;
    }
  }
  if (include_iter != Toks.end()) {
    Toks.erase(include_iter, include_iter + 3);
    put_this_after << "include \"qelib1.inc\";\n";
  }
  if (open_qasm_iter != Toks.end()) {
    Toks.erase(open_qasm_iter, open_qasm_iter + 3);
    sss << "OPENQASM 2.0;\n";
  }

  sss << put_this_after.str();
  sss << xx.str();

  std::map<std::string, int> creg_name_to_size;

  bool hasOracle = false;
  bool getOracleName = false;
  std::string oracleName;
  std::map<std::string, std::size_t> qreg_calls;
  std::vector<std::string> seen_ordered_qreg;

  for (int i = 0; i < Toks.size(); i++) {
    auto current_token = Toks[i];
    auto current_token_str = PP.getSpelling(current_token);

    if (current_token_str == "creg") {

      auto creg_name = PP.getSpelling(Toks[i + 1]);
      if (Toks[i+2].isNot(clang::tok::l_square)) {
          xacc::error("Must provide creg [ SIZE ].");
      }
      auto creg_size = PP.getSpelling(Toks[i + 3]);

      creg_name_to_size.insert({creg_name, std::stoi(creg_size)});
    }

    if (current_token_str == "measure") {
      // we have an ibm style measure,
      // so make sure that we map to individual measures
      // since we don't know the size of the qreg

      // next token is qreg name
      i++;
      current_token = Toks[i];
      current_token_str = PP.getSpelling(current_token);
      auto qreg_name = current_token_str;

      // next token could be [ or could be ->
      i++;
      current_token = Toks[i];
      if (current_token.is(clang::tok::l_square)) {
        i--;
        i--;
        current_token = Toks[i];
        while (current_token.isNot(clang::tok::semi)) {
          sss << PP.getSpelling(current_token) << " ";
          i++;
          current_token = Toks[i];
        }
        sss << ";\n";
        continue;

      } else {
        // the token is ->

        // the next one is the creg name
        i++;
        current_token = Toks[i];
        current_token_str = PP.getSpelling(current_token);
        auto creg_name = current_token_str;
        auto size = creg_name_to_size[creg_name];
        for (int k = 0; k < size; k++) {
          sss << "measure " << qreg_name << "[" << k << "] -> " << creg_name
              << "[" << k << "];\n";
        }
        i++;
        continue;
      }
    }

    if (current_token_str == "qreg") {

      // if the qasm string has qreg calls in it
      // then we need to add that as an allocation
      // in the re-written qrt code
      //   i++;
      //   current_token = ;

      // get qreg var name
      auto variable_name = PP.getSpelling(Toks[i + 1]);

      auto size = std::stoi(PP.getSpelling(Toks[i + 3]));
      qreg_calls.insert({variable_name, size});
      seen_ordered_qreg.push_back(variable_name);
    }

    if (getOracleName) {
      sss << " " << PP.getSpelling(current_token) << " ";
      oracleName = PP.getSpelling(current_token);
      getOracleName = false;
    } else if (PP.getSpelling(current_token) == "oracle") {
      sss << PP.getSpelling(current_token) << " ";
      getOracleName = true;
      hasOracle = true;
    } else if (current_token.is(tok::TokenKind::semi)) {
      sss << ";\n";
    } else if (current_token.is(tok::TokenKind::l_brace)) {
      // add space before lbrach
      sss << " " << PP.getSpelling(current_token) << " ";
    } else if (current_token.is(tok::TokenKind::r_brace)) {
      sss << " " << PP.getSpelling(current_token) << "\n";
    } else if (PP.getSpelling(current_token) == "creg" ||
               PP.getSpelling(current_token) == "qreg") {
      sss << PP.getSpelling(current_token) << " ";
    } else if (gates.count(PP.getSpelling(current_token)) ||
               PP.getSpelling(current_token) == oracleName) {
      sss << PP.getSpelling(current_token) << " ";
    } else {
      sss << PP.getSpelling(current_token) << " ";
    }
  }

  // std::cout << "FROM STAQ:\n" << sss.str() << "\n";

  auto compiler = xacc::getCompiler("staq");
  auto inst = compiler->compile(sss.str())->getComposites()[0];
  // std::cout << inst->toString() << "\n";

  // Map this CompositeInstruction to QRT calls
  auto visitor = std::make_shared<qcor::qrt_mapper>(inst->name());
  xacc::InstructionIterator iter(inst);
  while (iter.hasNext()) {
    auto next = iter.next();
    next->accept(visitor);
  }
  if (hasOracle) {
    ss << "auto anc = qalloc(" << 1000 << ");\n";
  }

  if (!qreg_calls.empty() && bufferNames.size() == qreg_calls.size()) {
    for (int i = 0; i < bufferNames.size(); i++) {
      if (!qreg_calls.count(bufferNames[i])) {
        // associate bufferName with seen_qreg[i];
        ss << "auto " << seen_ordered_qreg[i] << " = " << bufferNames[i]
           << ";\n";
      }
    }
  }
  ss << visitor->get_new_src();
}

} // namespace qcor
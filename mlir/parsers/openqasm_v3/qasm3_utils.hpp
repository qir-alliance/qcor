#pragma once
#include "qasm3BaseVisitor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <string> 
#include <vector>

namespace qcor {
// static std::vector<std::string> builtins{
//     "u3", "u2",   "u1",  "cx",  "id",  "u0",  "x",   "y",  "z",
//     "h",  "s",    "sdg", "t",   "tdg", "rx",  "ry",  "rz", "cz",
//     "cy", "swap", "ch",  "ccx", "crz", "cu1", "cu2", "cu3"};

template <class Op>
void split(const std::string& s, char delim, Op op) {
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, delim);) {
    *op++ = item;
  }
}

void printErrorMessage(const std::string msg, bool do_exit = true) ;

void printErrorMessage(const std::string msg, mlir::Value v);

void printErrorMessage(const std::string msg, std::vector<mlir::Value>&& v);

mlir::Location get_location(mlir::OpBuilder builder,
                            const std::string& file_name,
                            antlr4::ParserRuleContext* context) ;

inline std::vector<std::string> split(const std::string& s, char delim);
extern std::map<std::string, mlir::CmpIPredicate> antlr_to_mlir_predicate;



}
#pragma once
#include <string>
#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "qasm3BaseVisitor.h"

namespace qcor {

class ScopedSymbolTable;

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

void printErrorMessage(const std::string msg, bool do_exit = true);
void printErrorMessage(const std::string msg,
                       antlr4::ParserRuleContext* context, bool do_exit = true);
void printErrorMessage(const std::string msg,
                       antlr4::ParserRuleContext* context,
                       std::vector<mlir::Value>&& v, bool do_exit = true);
void printErrorMessage(const std::string msg, mlir::Value v);
void printErrorMessage(const std::string msg, std::vector<mlir::Value>&& v);

mlir::Location get_location(mlir::OpBuilder builder,
                            const std::string& file_name,
                            antlr4::ParserRuleContext* context);

std::vector<std::string> split(const std::string& s, char delim);

mlir::Type get_custom_opaque_type(const std::string& type,
                                  mlir::MLIRContext* context);

mlir::Value get_or_extract_qubit(const std::string& qreg_name,
                                 const std::size_t idx, mlir::Location location,
                                 ScopedSymbolTable& symbol_table,
                                 mlir::OpBuilder& builder);

mlir::Value get_or_create_constant_integer_value(
    const std::size_t idx, mlir::Location location, mlir::Type int_like_type,
    ScopedSymbolTable& symbol_table, mlir::OpBuilder& builder);

mlir::Value get_or_create_constant_index_value(const std::size_t idx,
                                               mlir::Location location,
                                               int width,
                                               ScopedSymbolTable& symbol_table,
                                               mlir::OpBuilder& builder);
extern std::map<std::string, mlir::CmpIPredicate> antlr_to_mlir_predicate;

}  // namespace qcor
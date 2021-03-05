#include "qasm3_utils.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "Quantum/QuantumOps.h"
#include "symbol_table.hpp"

namespace qcor {

void printErrorMessage(const std::string msg, bool do_exit) {
  std::cout << "\n[OPENQASM3 MLIRGen] Error\n" << msg << "\n\n";
  if (do_exit) exit(1);
}

void printErrorMessage(const std::string msg, mlir::Value v) {
  printErrorMessage(msg, false);
  v.dump();
  exit(1);
}

void printErrorMessage(const std::string msg, std::vector<mlir::Value>&& v) {
  printErrorMessage(msg, false);
  for (auto vv : v) vv.dump();
  exit(1);
}

mlir::Location get_location(mlir::OpBuilder builder,
                            const std::string& file_name,
                            antlr4::ParserRuleContext* context) {
  auto line = context->getStart()->getLine();
  auto col = context->getStart()->getCharPositionInLine();
  return builder.getFileLineColLoc(builder.getIdentifier(file_name), line, col);
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

mlir::Type get_custom_opaque_type(const std::string& type,
                                  mlir::MLIRContext* context) {
  llvm::StringRef type_name(type);
  mlir::Identifier dialect = mlir::Identifier::get("quantum", context);
  return mlir::OpaqueType::get(context, dialect, type_name);
}

mlir::Value get_or_extract_qubit(const std::string& qreg_name,
                                 const std::size_t idx, mlir::Location location,
                                 ScopedSymbolTable& symbol_table,
                                 mlir::OpBuilder& builder) {
  auto key = qreg_name + std::to_string(idx);
  if (symbol_table.has_symbol(key)) {
    return symbol_table.get_symbol(key);  // global_symbol_table[key];
  } else {
    auto qubits = symbol_table.get_symbol(qreg_name);
    mlir::Value pos = get_or_create_constant_integer_value(
        idx, location, builder.getI64Type(), symbol_table, builder);
    auto value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, get_custom_opaque_type("Qubit", builder.getContext()), qubits,
        pos);
    symbol_table.add_symbol(key, value);
    return value;
  }
}

mlir::Value get_or_create_constant_integer_value(
    const std::size_t idx, mlir::Location location, mlir::Type type,
    ScopedSymbolTable& symbol_table, mlir::OpBuilder& builder) {
  auto width = type.getIntOrFloatBitWidth();
  if (symbol_table.has_constant_integer(idx, width)) {
    return symbol_table.get_constant_integer(idx, width);
  } else {
    auto integer_attr = mlir::IntegerAttr::get(type, idx);
    auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
    symbol_table.add_constant_integer(idx, ret, width);
    return ret;
  }
}

mlir::Value get_or_create_constant_index_value(const std::size_t idx,
                                               mlir::Location location,
                                               int width,
                                               ScopedSymbolTable& symbol_table,
                                               mlir::OpBuilder& builder) {
  auto type = mlir::IntegerType::get(builder.getContext(), width);
  auto constant_int = get_or_create_constant_integer_value(
      idx, location, type, symbol_table, builder);
  return builder.create<mlir::IndexCastOp>(location, constant_int,
                                           builder.getIndexType());
}

std::map<std::string, mlir::CmpIPredicate> antlr_to_mlir_predicate{
    {"==", mlir::CmpIPredicate::eq},  {"!=", mlir::CmpIPredicate::ne},
    {"<=", mlir::CmpIPredicate::sle}, {">=", mlir::CmpIPredicate::sge},
    {"<", mlir::CmpIPredicate::slt},  {">", mlir::CmpIPredicate::sgt}};

}  // namespace qcor
#include "qasm3_utils.hpp"

#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "symbol_table.hpp"

namespace qcor {

void printErrorMessage(const std::string msg, bool do_exit) {
  std::cout << "\n[OPENQASM3 MLIRGen] Error\n" << msg << "\n\n";
  if (do_exit) exit(1);
}

void printErrorMessage(const std::string msg,
                       antlr4::ParserRuleContext* context, bool do_exit) {
  auto line = context->getStart()->getLine();
  auto col = context->getStart()->getCharPositionInLine();
  std::cout << "\n[OPENQASM3 MLIRGen] Error at " << line << ":" << col << "\n"
            << "   AntlrText: " << context->getText() << "\n"
            << "   " << msg << "\n\n";
  if (do_exit) exit(1);
}
void printErrorMessage(const std::string msg,
                       antlr4::ParserRuleContext* context,
                       std::vector<mlir::Value>&& v, bool do_exit) {
  auto line = context->getStart()->getLine();
  auto col = context->getStart()->getCharPositionInLine();
  std::cout << "\n[OPENQASM3 MLIRGen] Error at " << line << ":" << col << "\n"
            << "   AntlrText: " << context->getText() << "\n"
            << "   " << msg << "\n\n";
  std::cout << "MLIR Values:\n";
  for (auto vv : v) vv.dump();

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

mlir::Value get_or_extract_qubit(const std::string &qreg_name,
                                 const std::size_t idx, mlir::Location location,
                                 ScopedSymbolTable &symbol_table,
                                 mlir::OpBuilder &builder) {
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
    ScopedSymbolTable &symbol_table, mlir::OpBuilder &builder) {
  auto width = type.getIntOrFloatBitWidth();
  if (symbol_table.has_constant_integer(idx, width)) {
    return symbol_table.get_constant_integer(idx, width);
  } else {
    // Handle unsigned int constant:
    // ConstantOp (std dialect) doesn't support Signed type (uint)
    if (!type.cast<mlir::IntegerType>().isSignless()) {
      auto signless_int_type =
          builder.getIntegerType(type.getIntOrFloatBitWidth());
      auto integer_attr = mlir::IntegerAttr::get(signless_int_type, idx);

      auto ret =
          builder
              .create<mlir::quantum::IntegerCastOp>(
                  location, type,
                  builder.create<mlir::ConstantOp>(location, integer_attr))
              .output();
      symbol_table.add_constant_integer(idx, ret, width);
      return ret;
    } else {
      auto integer_attr = mlir::IntegerAttr::get(type, idx);
      assert(integer_attr.getType().cast<mlir::IntegerType>().isSignless());
      auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
      symbol_table.add_constant_integer(idx, ret, width);
      return ret;
    }
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

mlir::Type convertQasm3Type(qasm3::qasm3Parser::ClassicalTypeContext* ctx,
                            ScopedSymbolTable& symbol_table,
                            mlir::OpBuilder& builder, bool value_type) {
  auto type = ctx->getText();
  auto context = builder.getContext();
  llvm::StringRef qubit_type_name("Qubit"), array_type_name("Array"),
      result_type_name("Result");
  mlir::Identifier dialect = mlir::Identifier::get("quantum", context);
  auto qubit_type = mlir::OpaqueType::get(context, dialect, qubit_type_name);
  auto array_type = mlir::OpaqueType::get(context, dialect, array_type_name);
  auto result_type = mlir::IntegerType::get(context, 1);
  if (type == "bit") {
    // result type
    return result_type;
  } else if (type.find("bit") != std::string::npos &&
             type.find("[") != std::string::npos) {
    // array type
    auto start = type.find_first_of("[");
    auto finish = type.find_first_of("]");
    auto idx_str = type.substr(start + 1, finish - start - 1);
    auto bit_size = symbol_table.evaluate_constant_integer_expression(idx_str);

    mlir::Type mlir_type;
    llvm::ArrayRef<int64_t> shaperef(bit_size);
    mlir_type = mlir::MemRefType::get(shaperef, result_type);
    return mlir_type;
  } else if (type == "bool") {
    mlir::Type mlir_type;
    llvm::ArrayRef<int64_t> shaperef{};
    mlir_type = mlir::MemRefType::get(shaperef, builder.getIntegerType(1));
    return mlir_type;
  } else if (type.find("uint") != std::string::npos) {
    auto start = type.find_first_of("[");
    auto finish = type.find_first_of("]");
    auto idx_str = type.substr(start + 1, finish - start - 1);
    auto bit_size = symbol_table.evaluate_constant_integer_expression(idx_str);

    mlir::Type mlir_type;
    llvm::ArrayRef<int64_t> shaperef{};
    mlir_type = mlir::MemRefType::get(shaperef,
                                      builder.getIntegerType(bit_size, false));
    return mlir_type;
  } else if (type.find("int") != std::string::npos) {
    auto start = type.find_first_of("[");
    int64_t bit_size;
    if (start == std::string::npos) {
      bit_size = 32;
    } else {
      auto finish = type.find_first_of("]");
      auto idx_str = type.substr(start + 1, finish - start - 1);
      bit_size = symbol_table.evaluate_constant_integer_expression(idx_str);
    }

    mlir::Type mlir_type;
    if (value_type) {
      return builder.getIntegerType(bit_size);
    }
    llvm::ArrayRef<int64_t> shaperef{};
    mlir_type =
        mlir::MemRefType::get(shaperef, builder.getIntegerType(bit_size));
    return mlir_type;
  } else if (type.find("float") != std::string::npos) {
    auto start = type.find_first_of("[");
    int64_t bit_size;
    if (start == std::string::npos) {
      bit_size = 32;
    } else {
      auto finish = type.find_first_of("]");
      auto idx_str = type.substr(start + 1, finish - start - 1);
      bit_size = symbol_table.evaluate_constant_integer_expression(idx_str);
    }

    mlir::Type mlir_type;
    if (bit_size == 16) {
      mlir_type = builder.getF16Type();
    } else if (bit_size == 32) {
      mlir_type = builder.getF32Type();
    } else if (bit_size == 64) {
      mlir_type = builder.getF64Type();
    } else {
      printErrorMessage("We only accept float types of bit width 16, 32, 64.");
    }

    if (value_type) {
      return mlir_type;
    }
    llvm::ArrayRef<int64_t> shaperef{};
    mlir_type = mlir::MemRefType::get(shaperef, mlir_type);
    return mlir_type;
  } else if (type.find("double") != std::string::npos) {
    int64_t bit_size = 64;
    mlir::Type mlir_type;
    if (bit_size == 16) {
      mlir_type = builder.getF16Type();
    } else if (bit_size == 32) {
      mlir_type = builder.getF32Type();
    } else if (bit_size == 64) {
      mlir_type = builder.getF64Type();
    } else {
      printErrorMessage("We only accept float types of bit width 16, 32, 64.");
    }

    if (value_type) {
      return mlir_type;
    }
    llvm::ArrayRef<int64_t> shaperef{};
    mlir_type = mlir::MemRefType::get(shaperef, mlir_type);
    return mlir_type;
  } else if (type.find("angle") != std::string::npos) {
  }

  // arg_names.push_back(arg->association()->Identifier()->getText());
  // arg_attributes.push_back({});
  printErrorMessage("Could not convert qasm3 type to mlir type.", ctx);
  return mlir::Type();
}

mlir::Value cast_array_index_value_if_required(mlir::Type array_type, mlir::Value raw_index, mlir::Location location, mlir::OpBuilder &builder) {
  // Memref must use index type
  if (array_type.isa<mlir::MemRefType>() &&
      !raw_index.getType().isa<mlir::IndexType>()) {
    return builder.create<mlir::IndexCastOp>(location, builder.getIndexType(),
                                             raw_index);
  }
  // QIR arrays: must use I64
  if (array_type.isa<mlir::OpaqueType>() &&
      array_type.cast<mlir::OpaqueType>().getTypeData().str() == "Array" &&
      raw_index.getType().isa<mlir::IndexType>()) {
    return builder.create<mlir::IndexCastOp>(location, builder.getI64Type(),
                                             raw_index);
  }

  // No need to do anything
  return raw_index;
}


std::map<std::string, mlir::CmpIPredicate> antlr_to_mlir_predicate{
    {"==", mlir::CmpIPredicate::eq},  {"!=", mlir::CmpIPredicate::ne},
    {"<=", mlir::CmpIPredicate::sle}, {">=", mlir::CmpIPredicate::sge},
    {"<", mlir::CmpIPredicate::slt},  {">", mlir::CmpIPredicate::sgt}};

std::map<std::string, mlir::CmpFPredicate> antlr_to_mlir_fpredicate{
    {"==", mlir::CmpFPredicate::OEQ}, {"!=", mlir::CmpFPredicate::ONE},
    {"<=", mlir::CmpFPredicate::OLE}, {">=", mlir::CmpFPredicate::OGE},
    {"<", mlir::CmpFPredicate::OLT},  {">", mlir::CmpFPredicate::OGT}};
}  // namespace qcor
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace quantum {
class InstOp;
class QallocOp;
class DeallocOp;
class ExtractQubitOp;
class QRTInitOp;
class QRTFinalizeOp;
}  // namespace quantum
}  // namespace mlir

namespace mlir {
namespace quantum {

class QuantumDialect : public mlir::Dialect {
 public:
  explicit QuantumDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "quantum"; }
};
class ExtractQubitOpAdaptor {
 public:
  ExtractQubitOpAdaptor(::mlir::ValueRange values,
                        ::mlir::DictionaryAttr attrs = nullptr);
  ExtractQubitOpAdaptor(ExtractQubitOp &op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value qreg();
  ::mlir::Value idx();
  ::mlir::LogicalResult verify(::mlir::Location loc);

 private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ExtractQubitOp
    : public ::mlir::Op<ExtractQubitOp, ::mlir::OpTrait::ZeroRegion,
                        ::mlir::OpTrait::OneResult,
                        ::mlir::OpTrait::ZeroSuccessor,
                        ::mlir::OpTrait::NOperands<2>::Impl> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractQubitOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value qreg();
  ::mlir::Value idx();
  ::mlir::MutableOperandRange qregMutable();
  ::mlir::MutableOperandRange idxMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value qbit();
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type qbit,
                    ::mlir::Value qreg, ::mlir::Value idx);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value qreg,
                    ::mlir::Value idx);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
}  // namespace quantum
}  // namespace mlir
namespace mlir {
namespace quantum {

//===----------------------------------------------------------------------===//
// ::mlir::quantum::InstOp declarations
//===----------------------------------------------------------------------===//

class InstOpAdaptor {
 public:
  InstOpAdaptor(::mlir::ValueRange values,
                ::mlir::DictionaryAttr attrs = nullptr);
  InstOpAdaptor(InstOp &op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange qubits();
  ::mlir::StringAttr name();
  ::mlir::DenseElementsAttr params();
  ::mlir::LogicalResult verify(::mlir::Location loc);

 private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class InstOp
    : public ::mlir::Op<
          InstOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults,
          ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = InstOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range qubits();
  ::mlir::MutableOperandRange qubitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value bit();
  ::mlir::StringAttr nameAttr();
  ::llvm::StringRef name();
  ::mlir::DenseElementsAttr paramsAttr();
  ::llvm::Optional<::mlir::DenseElementsAttr> params();
  void nameAttr(::mlir::StringAttr attr);
  void paramsAttr(::mlir::DenseElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    /*optional*/ ::mlir::Type bit, ::mlir::StringAttr name,
                    ::mlir::ValueRange qubits,
                    /*optional*/ ::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::StringAttr name,
                    ::mlir::ValueRange qubits,
                    /*optional*/ ::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    /*optional*/ ::mlir::Type bit, ::llvm::StringRef name,
                    ::mlir::ValueRange qubits,
                    /*optional*/ ::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::llvm::StringRef name,
                    ::mlir::ValueRange qubits,
                    /*optional*/ ::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
}  // namespace quantum
}  // namespace mlir
namespace mlir {
namespace quantum {

//===----------------------------------------------------------------------===//
// ::mlir::quantum::QallocOp declarations
//===----------------------------------------------------------------------===//
class QRTInitOpAdaptor {
 public:
  QRTInitOpAdaptor(::mlir::ValueRange values,
                   ::mlir::DictionaryAttr attrs = nullptr);
  QRTInitOpAdaptor(QRTInitOp &op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value argc();
  ::mlir::Value argv();
  ::mlir::LogicalResult verify(::mlir::Location loc);

 private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class QRTInitOp
    : public ::mlir::Op<
          QRTInitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult,
          ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = QRTInitOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value argc();
  ::mlir::Value argv();
  ::mlir::MutableOperandRange argcMutable();
  ::mlir::MutableOperandRange argvMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Value argc,
                    ::mlir::Value argv);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value argc,
                    ::mlir::Value argv);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
class QRTFinalizeOpAdaptor {
 public:
  QRTFinalizeOpAdaptor(::mlir::ValueRange values,
                       ::mlir::DictionaryAttr attrs = nullptr);
  QRTFinalizeOpAdaptor(QRTFinalizeOp &op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::LogicalResult verify(::mlir::Location loc);

 private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class QRTFinalizeOp
    : public ::mlir::Op<QRTFinalizeOp, ::mlir::OpTrait::ZeroRegion,
                        ::mlir::OpTrait::ZeroResult,
                        ::mlir::OpTrait::ZeroSuccessor,
                        ::mlir::OpTrait::ZeroOperands> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = QRTFinalizeOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
class QallocOpAdaptor {
 public:
  QallocOpAdaptor(::mlir::ValueRange values,
                  ::mlir::DictionaryAttr attrs = nullptr);
  QallocOpAdaptor(QallocOp &op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::IntegerAttr size();
  ::mlir::StringAttr name();
  ::mlir::LogicalResult verify(::mlir::Location loc);

 private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class QallocOp
    : public ::mlir::Op<
          QallocOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = QallocOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value qubits();
  ::mlir::IntegerAttr sizeAttr();
  ::llvm::APInt size();
  ::mlir::StringAttr nameAttr();
  ::llvm::StringRef name();
  void sizeAttr(::mlir::IntegerAttr attr);
  void nameAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type qubits,
                    ::mlir::IntegerAttr size, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr size,
                    ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type qubits,
                    ::mlir::IntegerAttr size, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr size,
                    ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
};
class DeallocOpAdaptor {
 public:
  DeallocOpAdaptor(::mlir::ValueRange values,
                   ::mlir::DictionaryAttr attrs = nullptr);
  DeallocOpAdaptor(DeallocOp &op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value qubits();
  ::mlir::LogicalResult verify(::mlir::Location loc);

 private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class DeallocOp
    : public ::mlir::Op<
          DeallocOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult,
          ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeallocOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value qubits();
  ::mlir::MutableOperandRange qubitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Value qubits);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value qubits);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
}  // namespace quantum
}  // namespace mlir

/* Used below to generate the above code
mlir-tblgen -gen-op-decls Ops.td -I ../../mlir-llvm/mlir/include
mlir-tblgen -gen-op-defs Ops.td -I ../../mlir-llvm/mlir/include

def QuantumDialect : Dialect {
    let name = "quantum";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::quantum";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//


def InstOp : Op<QuantumDialect, "inst", []> {
   let arguments = (ins StrAttr:$name, StringElementsAttr:$qreg_names,
IndexElementsAttr:$qubits, F64ElementsAttr:$params); let results = (outs);
}
*/
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace quantum {
class InstOp;
class QallocOp;
class QInstOp;
class ExtractQubitOp;
}  // namespace quantum
}  // namespace mlir

namespace mlir {
namespace quantum {
// struct QubitTypeStorage : public TypeStorage {
//   QubitTypeStorage(std::int64_t _qubit_idx, std::string _enclosed_register)
//       : qubit_idx(_qubit_idx), enclosed_register(_enclosed_register) {}

//   /// The hash key for this storage is a pair of the integer and type params.
//   using KeyTy = std::pair<std::int64_t, std::string>;

//   /// Define the comparison function for the key type.
//   bool operator==(const KeyTy &key) const {
//     return key == KeyTy(qubit_idx, enclosed_register);
//   }

//   /// Define a hash function for the key type.
//   /// Note: This isn't necessary because std::pair, unsigned, and Type all have
//   /// hash functions already available.
//   static llvm::hash_code hashKey(const KeyTy &key) {
//     return llvm::hash_combine(key.first, key.second);
//   }

//   /// Define a construction function for the key type.
//   /// Note: This isn't necessary because KeyTy can be directly constructed with
//   /// the given parameters.
//   static KeyTy getKey(std::int64_t _qubit_idx, std::string enc_reg) {
//     return KeyTy(_qubit_idx, enc_reg);
//   }

//   /// Define a construction method for creating a new instance of this storage.
//   static QubitTypeStorage *construct(TypeStorageAllocator &allocator,
//                                      const KeyTy &key) {
//     return new (allocator.allocate<QubitTypeStorage>())
//         QubitTypeStorage(key.first, key.second);
//   }

//   /// The parametric data held by the storage class.
//   std::int64_t qubit_idx;
//   std::string enclosed_register;
// };

// class QubitType : public Type::TypeBase<QubitType, Type, QubitTypeStorage> {
//  public:
//   /// Inherit some necessary constructors from 'TypeBase'.
//   using Base::Base;

//   /// This method is used to get an instance of the 'ComplexType'. This method
//   /// asserts that all of the construction invariants were satisfied. To
//   /// gracefully handle failed construction, getChecked should be used instead.
//   static QubitType get(mlir::MLIRContext* ctx, int64_t qbit, std::string enc_reg) {
//     // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
//     // of this type. All parameters to the storage class are passed after the
//     // context.
//     return Base::get(ctx, qbit, enc_reg);
//   }

//   /// This method is used to get an instance of the 'ComplexType', defined at
//   /// the given location. If any of the construction invariants are invalid,
//   /// errors are emitted with the provided location and a null type is returned.
//   /// Note: This method is completely optional.
//   static QubitType getChecked(std::int64_t qbit, std::string enc_reg, Location location) {
//     // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
//     // instance of this type. All parameters to the storage class are passed
//     // after the location.
//     return Base::getChecked(location, qbit, enc_reg);
//   }

//   /// This method is used to verify the construction invariants passed into the
//   /// 'get' and 'getChecked' methods. Note: This method is completely optional.
//   static LogicalResult verifyConstructionInvariants(Location loc,
//                                                     std::int64_t qbit, std::string enc_reg) {
//     // Our type only allows non-zero parameters.
//     if (qbit < 0)
//       return emitError(loc) << "non-zero parameter passed to 'QubitType'";
//     return success();
//   }

//   /// Return the parameter value.
//   std::int64_t getQubitIndex() {
//     // 'getImpl' returns a pointer to our internal storage instance.
//     return getImpl()->qubit_idx;
//   }

//   /// Return the integer parameter type.
//   std::string getEnclosedRegister() {
//     // 'getImpl' returns a pointer to our internal storage instance.
//     return getImpl()->enclosed_register;
//   }
// };

class QuantumDialect : public mlir::Dialect {
 public:
  explicit QuantumDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "quantum"; }
};
class ExtractQubitOpAdaptor {
public:
  ExtractQubitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  ExtractQubitOpAdaptor(ExtractQubitOp& op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value qreg();
  ::mlir::Value idx();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ExtractQubitOp : public ::mlir::Op<ExtractQubitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type qbit, ::mlir::Value qreg, ::mlir::Value idx);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value qreg, ::mlir::Value idx);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace quantum
} // namespace mlir
namespace mlir {
namespace quantum {

//===----------------------------------------------------------------------===//
// ::mlir::quantum::InstOp declarations
//===----------------------------------------------------------------------===//

class InstOpAdaptor {
public:
  InstOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  InstOpAdaptor(InstOp&op);
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
class InstOp : public ::mlir::Op<InstOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands> {
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
  ::llvm::Optional< ::mlir::DenseElementsAttr > params();
  void nameAttr(::mlir::StringAttr attr);
  void paramsAttr(::mlir::DenseElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type bit, ::mlir::StringAttr name, ::mlir::ValueRange qubits, /*optional*/::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange qubits, /*optional*/::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type bit, ::llvm::StringRef name, ::mlir::ValueRange qubits, /*optional*/::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange qubits, /*optional*/::mlir::DenseElementsAttr params);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace quantum
} // namespace mlir
namespace mlir {
namespace quantum {

//===----------------------------------------------------------------------===//
// ::mlir::quantum::QallocOp declarations
//===----------------------------------------------------------------------===//

class QallocOpAdaptor {
public:
  QallocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  QallocOpAdaptor(QallocOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::IntegerAttr size();
  ::mlir::StringAttr name();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class QallocOp : public ::mlir::Op<QallocOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type qubits, ::mlir::IntegerAttr size, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr size, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type qubits, ::mlir::IntegerAttr size, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr size, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
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
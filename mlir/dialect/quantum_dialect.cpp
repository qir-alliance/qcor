#include "quantum_dialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::quantum;

namespace mlir {
namespace quantum {
bool isOpaqueTypeWithName(mlir::Type type, std::string dialect,
                          std::string type_name) {
  if (type.isa<OpaqueType>() && dialect == "quantum") {
    if (type_name == "Qubit") {
      return true;
    }
    if (type_name == "Result") {
      return true;
    }
  }

  return false;
}
QuantumDialect::QuantumDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<QuantumDialect>()) {
  addOperations<InstOp, QallocOp, ExtractQubitOp, DeallocOp, QRTInitOp,
                QRTFinalizeOp, SetQregOp>();
}
QRTFinalizeOpAdaptor::QRTFinalizeOpAdaptor(::mlir::ValueRange values,
                                           ::mlir::DictionaryAttr attrs)
    : odsOperands(values), odsAttrs(attrs) {}

QRTFinalizeOpAdaptor::QRTFinalizeOpAdaptor(QRTFinalizeOp &op)
    : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {}

std::pair<unsigned, unsigned> QRTFinalizeOpAdaptor::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::ValueRange QRTFinalizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult QRTFinalizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef QRTFinalizeOp::getOperationName() {
  return "quantum.finalize";
}

std::pair<unsigned, unsigned> QRTFinalizeOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range QRTFinalizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> QRTFinalizeOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range QRTFinalizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void QRTFinalizeOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState) {}

void QRTFinalizeOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState,
                          ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void QRTFinalizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                          ::mlir::TypeRange resultTypes,
                          ::mlir::ValueRange operands,
                          ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult QRTFinalizeOp::verify() {
  if (failed(QRTFinalizeOpAdaptor(*this).verify(this->getLoc())))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
  }
  {
    unsigned index = 0;
    (void)index;
  }
  return ::mlir::success();
}
ExtractQubitOpAdaptor::ExtractQubitOpAdaptor(::mlir::ValueRange values,
                                             ::mlir::DictionaryAttr attrs)
    : odsOperands(values), odsAttrs(attrs) {}

ExtractQubitOpAdaptor::ExtractQubitOpAdaptor(ExtractQubitOp &op)
    : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {}

std::pair<unsigned, unsigned>
ExtractQubitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExtractQubitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractQubitOpAdaptor::qreg() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractQubitOpAdaptor::idx() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult ExtractQubitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ExtractQubitOp::getOperationName() {
  return "quantum.qextract";
}

std::pair<unsigned, unsigned> ExtractQubitOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractQubitOp::getODSOperands(
    unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Value ExtractQubitOp::qreg() { return *getODSOperands(0).begin(); }

::mlir::Value ExtractQubitOp::idx() { return *getODSOperands(1).begin(); }

::mlir::MutableOperandRange ExtractQubitOp::qregMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ExtractQubitOp::idxMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ExtractQubitOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractQubitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Value ExtractQubitOp::qbit() { return *getODSResults(0).begin(); }

void ExtractQubitOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState, ::mlir::Type qbit,
                           ::mlir::Value qreg, ::mlir::Value idx) {
  odsState.addOperands(qreg);
  odsState.addOperands(idx);
  odsState.addTypes(qbit);
}

void ExtractQubitOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState,
                           ::mlir::TypeRange resultTypes, ::mlir::Value qreg,
                           ::mlir::Value idx) {
  odsState.addOperands(qreg);
  odsState.addOperands(idx);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractQubitOp::build(
    ::mlir::OpBuilder &, ::mlir::OperationState &odsState,
    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractQubitOp::verify() {
  if (failed(ExtractQubitOpAdaptor(*this).verify(this->getLoc())))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::VectorType>())) &&
            ((isOpaqueTypeWithName(
                v.getType().cast<::mlir::ShapedType>().getElementType(),
                "quantum", "Qubit"))))) {
        return emitOpError("operand #")
               << index
               << " must be vector of opaque qubit type values, but got "
               << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isSignlessInteger(64)))) {
        return emitOpError("operand #")
               << index << " must be 64-bit signless integer, but got "
               << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((isOpaqueTypeWithName(v.getType(), "quantum", "Qubit")))) {
        return emitOpError("result #")
               << index << " must be opaque qubit type, but got "
               << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

}  // namespace quantum
}  // namespace mlir
namespace mlir {
namespace quantum {

//===----------------------------------------------------------------------===//
// ::mlir::quantum::InstOp definitions
//===----------------------------------------------------------------------===//

InstOpAdaptor::InstOpAdaptor(::mlir::ValueRange values,
                             ::mlir::DictionaryAttr attrs)
    : odsOperands(values), odsAttrs(attrs) {}

InstOpAdaptor::InstOpAdaptor(InstOp &op)
    : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {}

std::pair<unsigned, unsigned> InstOpAdaptor::getODSOperandIndexAndLength(
    unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value
  // count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static
  // variadic operand, we need to offset by (variadicSize - 1) to get where the
  // dynamic value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange InstOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange InstOpAdaptor::qubits() { return getODSOperands(0); }

::mlir::StringAttr InstOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::DenseElementsAttr InstOpAdaptor::params() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseElementsAttr attr =
      odsAttrs.get("params").dyn_cast_or_null<::mlir::DenseElementsAttr>();
  return attr;
}

::mlir::LogicalResult InstOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_name = odsAttrs.get("name");
    if (!tblgen_name)
      return emitError(loc,
                       "'quantum.inst' op "
                       "requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>())))
      return emitError(
          loc,
          "'quantum.inst' op "
          "attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_params = odsAttrs.get("params");
    if (tblgen_params) {
      if (!((tblgen_params.isa<::mlir::DenseFPElementsAttr>() &&
             tblgen_params.cast<::mlir::DenseElementsAttr>()
                 .getType()
                 .getElementType()
                 .isF64())))
        return emitError(loc,
                         "'quantum.inst' op "
                         "attribute 'params' failed to satisfy constraint: "
                         "64-bit float elements attribute");
    }
  }
  return ::mlir::success();
}

::llvm::StringRef InstOp::getOperationName() { return "quantum.inst"; }

std::pair<unsigned, unsigned> InstOp::getODSOperandIndexAndLength(
    unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value
  // count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static
  // variadic operand, we need to offset by (variadicSize - 1) to get where the
  // dynamic value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range InstOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InstOp::qubits() { return getODSOperands(0); }

::mlir::MutableOperandRange InstOp::qubitsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> InstOp::getODSResultIndexAndLength(
    unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value
  // count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static
  // variadic operand, we need to offset by (variadicSize - 1) to get where the
  // dynamic value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range InstOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Value InstOp::bit() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::StringAttr InstOp::nameAttr() {
  return this->getAttr("name").cast<::mlir::StringAttr>();
}

::llvm::StringRef InstOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::DenseElementsAttr InstOp::paramsAttr() {
  return this->getAttr("params").dyn_cast_or_null<::mlir::DenseElementsAttr>();
}

::llvm::Optional<::mlir::DenseElementsAttr> InstOp::params() {
  auto attr = paramsAttr();
  return attr ? ::llvm::Optional<::mlir::DenseElementsAttr>(attr)
              : (::llvm::None);
}

void InstOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void InstOp::paramsAttr(::mlir::DenseElementsAttr attr) {
  (*this)->setAttr("params", attr);
}

void InstOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState,
                   /*optional*/ ::mlir::Type bit, ::mlir::StringAttr name,
                   ::mlir::ValueRange qubits,
                   /*optional*/ ::mlir::DenseElementsAttr params) {
  odsState.addOperands(qubits);
  odsState.addAttribute("name", name);
  if (params) {
    odsState.addAttribute("params", params);
  }
  if (bit) odsState.addTypes(bit);
}

void InstOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState,
                   ::mlir::TypeRange resultTypes, ::mlir::StringAttr name,
                   ::mlir::ValueRange qubits,
                   /*optional*/ ::mlir::DenseElementsAttr params) {
  odsState.addOperands(qubits);
  odsState.addAttribute("name", name);
  if (params) {
    odsState.addAttribute("params", params);
  }
  odsState.addTypes(resultTypes);
}

void InstOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState,
                   /*optional*/ ::mlir::Type bit, ::llvm::StringRef name,
                   ::mlir::ValueRange qubits,
                   /*optional*/ ::mlir::DenseElementsAttr params) {
  odsState.addOperands(qubits);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (params) {
    odsState.addAttribute("params", params);
  }
  if (bit) odsState.addTypes(bit);
}

void InstOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState,
                   ::mlir::TypeRange resultTypes, ::llvm::StringRef name,
                   ::mlir::ValueRange qubits,
                   /*optional*/ ::mlir::DenseElementsAttr params) {
  odsState.addOperands(qubits);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (params) {
    odsState.addAttribute("params", params);
  }
  odsState.addTypes(resultTypes);
}

void InstOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                   ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                   ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InstOp::verify() {
  if (failed(InstOpAdaptor(*this).verify(this->getLoc())))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((isOpaqueTypeWithName(v.getType(), "quantum", "Qubit")))) {
        return emitOpError("operand #")
               << index << " must be opaque qubit type, but got "
               << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);
    if (valueGroup0.size() > 1)
      return emitOpError("result group starting at #")
             << index << " requires 0 or 1 element, but found "
             << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((isOpaqueTypeWithName(v.getType(), "quantum", "Result")))) {
        return emitOpError("result #")
               << index << " must be opaque result type, but got "
               << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

QRTInitOpAdaptor::QRTInitOpAdaptor(::mlir::ValueRange values,
                                   ::mlir::DictionaryAttr attrs)
    : odsOperands(values), odsAttrs(attrs) {}

QRTInitOpAdaptor::QRTInitOpAdaptor(QRTInitOp &op)
    : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {}

std::pair<unsigned, unsigned> QRTInitOpAdaptor::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::ValueRange QRTInitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value QRTInitOpAdaptor::argc() { return *getODSOperands(0).begin(); }

::mlir::Value QRTInitOpAdaptor::argv() { return *getODSOperands(1).begin(); }

::mlir::LogicalResult QRTInitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef QRTInitOp::getOperationName() { return "quantum.init"; }

std::pair<unsigned, unsigned> QRTInitOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range QRTInitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Value QRTInitOp::argc() { return *getODSOperands(0).begin(); }

::mlir::Value QRTInitOp::argv() { return *getODSOperands(1).begin(); }

::mlir::MutableOperandRange QRTInitOp::argcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange QRTInitOp::argvMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> QRTInitOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range QRTInitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void QRTInitOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, ::mlir::Value argc,
                      ::mlir::Value argv) {
  odsState.addOperands(argc);
  odsState.addOperands(argv);
}

void QRTInitOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      ::mlir::TypeRange resultTypes, ::mlir::Value argc,
                      ::mlir::Value argv) {
  odsState.addOperands(argc);
  odsState.addOperands(argv);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void QRTInitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                      ::mlir::TypeRange resultTypes,
                      ::mlir::ValueRange operands,
                      ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult QRTInitOp::verify() {
  if (failed(QRTInitOpAdaptor(*this).verify(this->getLoc())))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isInteger(32)))) {
        return emitOpError("operand #")
               << index << " must be 32-bit integer, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((isOpaqueTypeWithName(v.getType(), "quantum", "ArgvType")))) {
        return emitOpError("operand #")
               << index << " must be opaque argv type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0;
    (void)index;
  }
  return ::mlir::success();
}

}  // namespace quantum
}  // namespace mlir
namespace mlir {
namespace quantum {

//===----------------------------------------------------------------------===//
// ::mlir::quantum::QallocOp definitions
//===----------------------------------------------------------------------===//

QallocOpAdaptor::QallocOpAdaptor(::mlir::ValueRange values,
                                 ::mlir::DictionaryAttr attrs)
    : odsOperands(values), odsAttrs(attrs) {}

QallocOpAdaptor::QallocOpAdaptor(QallocOp &op)
    : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {}

std::pair<unsigned, unsigned> QallocOpAdaptor::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::ValueRange QallocOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr QallocOpAdaptor::size() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("size").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::StringAttr QallocOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult QallocOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_size = odsAttrs.get("size");
    if (!tblgen_size)
      return emitError(loc,
                       "'quantum.qalloc' op "
                       "requires attribute 'size'");
    if (!(((tblgen_size.isa<::mlir::IntegerAttr>())) &&
          ((tblgen_size.cast<::mlir::IntegerAttr>().getType().isInteger(64)))))
      return emitError(loc,
                       "'quantum.qalloc' op "
                       "attribute 'size' failed to satisfy constraint: 64-bit "
                       "integer attribute");
  }
  {
    auto tblgen_name = odsAttrs.get("name");
    if (!tblgen_name)
      return emitError(loc,
                       "'quantum.qalloc' op "
                       "requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>())))
      return emitError(
          loc,
          "'quantum.qalloc' op "
          "attribute 'name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef QallocOp::getOperationName() { return "quantum.qalloc"; }

std::pair<unsigned, unsigned> QallocOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range QallocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> QallocOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range QallocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Value QallocOp::qubits() { return *getODSResults(0).begin(); }

::mlir::IntegerAttr QallocOp::sizeAttr() {
  return this->getAttr("size").cast<::mlir::IntegerAttr>();
}

::llvm::APInt QallocOp::size() {
  auto attr = sizeAttr();
  return attr.getValue();
}

::mlir::StringAttr QallocOp::nameAttr() {
  return this->getAttr("name").cast<::mlir::StringAttr>();
}

::llvm::StringRef QallocOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

void QallocOp::sizeAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("size", attr);
}

void QallocOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void QallocOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState, ::mlir::Type qubits,
                     ::mlir::IntegerAttr size, ::mlir::StringAttr name) {
  odsState.addAttribute("size", size);
  odsState.addAttribute("name", name);
  odsState.addTypes(qubits);
}

void QallocOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState,
                     ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr size,
                     ::mlir::StringAttr name) {
  odsState.addAttribute("size", size);
  odsState.addAttribute("name", name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void QallocOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState, ::mlir::Type qubits,
                     ::mlir::IntegerAttr size, ::llvm::StringRef name) {
  odsState.addAttribute("size", size);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  odsState.addTypes(qubits);
}

void QallocOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState,
                     ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr size,
                     ::llvm::StringRef name) {
  odsState.addAttribute("size", size);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void QallocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                     ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                     ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult QallocOp::verify() {
  if (failed(QallocOpAdaptor(*this).verify(this->getLoc())))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::VectorType>())) &&
            ((isOpaqueTypeWithName(
                v.getType().cast<::mlir::ShapedType>().getElementType(),
                "quantum", "Qubit"))))) {
        return emitOpError("result #")
               << index
               << " must be vector of opaque qubit type values, but got "
               << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

DeallocOpAdaptor::DeallocOpAdaptor(::mlir::ValueRange values,
                                   ::mlir::DictionaryAttr attrs)
    : odsOperands(values), odsAttrs(attrs) {}

DeallocOpAdaptor::DeallocOpAdaptor(DeallocOp &op)
    : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {}

std::pair<unsigned, unsigned> DeallocOpAdaptor::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DeallocOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeallocOpAdaptor::qubits() { return *getODSOperands(0).begin(); }

::mlir::LogicalResult DeallocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef DeallocOp::getOperationName() { return "quantum.dealloc"; }

std::pair<unsigned, unsigned> DeallocOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DeallocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Value DeallocOp::qubits() { return *getODSOperands(0).begin(); }

::mlir::MutableOperandRange DeallocOp::qubitsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DeallocOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DeallocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, ::mlir::Value qubits) {
  odsState.addOperands(qubits);
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      ::mlir::TypeRange resultTypes, ::mlir::Value qubits) {
  odsState.addOperands(qubits);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeallocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                      ::mlir::TypeRange resultTypes,
                      ::mlir::ValueRange operands,
                      ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeallocOp::verify() {
  if (failed(DeallocOpAdaptor(*this).verify(this->getLoc())))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((isOpaqueTypeWithName(v.getType(), "quantum", "Array")))) {
        return emitOpError("operand #")
               << index << " must be opaque array type, but got "
               << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0;
    (void)index;
  }
  return ::mlir::success();
}

static mlir::ParseResult parseQallocOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  // SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
  // llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (  // parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  // if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
  //   if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
  //                              result.operands))
  //     return mlir::failure();
  //   result.addTypes(funcType.getResults());
  //   return mlir::success();
  // }

  // Otherwise, the parsed type is the type of both operands and results.
  // if (parser.resolveOperands(operands, type, result.operands))
  //   return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}
::mlir::ParseResult QallocOp::parse(::mlir::OpAsmParser &parser,
                                    ::mlir::OperationState &result) {
  return ::parseQallocOp(parser, result);
}

//===----------------------------------------------------------------------===//
// ::mlir::quantum::SetQregOp definitions
//===----------------------------------------------------------------------===//

SetQregOpAdaptor::SetQregOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SetQregOpAdaptor::SetQregOpAdaptor(SetQregOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SetQregOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SetQregOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SetQregOpAdaptor::qreg() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult SetQregOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SetQregOp::getOperationName() {
  return "quantum.set_qreg";
}

std::pair<unsigned, unsigned> SetQregOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SetQregOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SetQregOp::qreg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SetQregOp::qregMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SetQregOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SetQregOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SetQregOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value qreg) {
  odsState.addOperands(qreg);
}

void SetQregOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value qreg) {
  odsState.addOperands(qreg);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SetQregOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SetQregOp::verify() {
  if (failed(SetQregOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((isOpaqueTypeWithName(v.getType(), "quantum", "QregType")))) {
        return emitOpError("operand #") << index << " must be opaque qreg type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}
}  // namespace quantum
}  // namespace mlir

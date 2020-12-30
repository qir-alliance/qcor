#include "quantum_dialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::quantum;

namespace mlir {
namespace quantum {
QuantumDialect::QuantumDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<QuantumDialect>()) {
  addOperations<InstOp, QallocOp, ReturnOp>();
}
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
  return {index, 1};
}

::mlir::Operation::result_range InstOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
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
                   ::mlir::OperationState &odsState, ::mlir::StringAttr name,
                   ::mlir::ValueRange qubits,
                   /*optional*/ ::mlir::DenseElementsAttr params) {
  odsState.addOperands(qubits);
  odsState.addAttribute("name", name);
  if (params) {
    odsState.addAttribute("params", params);
  }
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
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InstOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState, ::llvm::StringRef name,
                   ::mlir::ValueRange qubits,
                   /*optional*/ ::mlir::DenseElementsAttr params) {
  odsState.addOperands(qubits);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (params) {
    odsState.addAttribute("params", params);
  }
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
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InstOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                   ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                   ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
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
            ((v.getType()
                  .cast<::mlir::ShapedType>()
                  .getElementType()
                  .isSignlessInteger(64))))) {
        return emitOpError("result #")
               << index
               << " must be vector of 64-bit signless integer values, but got "
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
// ::mlir::quantum::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values,
                                 ::mlir::DictionaryAttr attrs)
    : odsOperands(values), odsAttrs(attrs) {}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp &op)
    : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(
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

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::input() { return getODSOperands(0); }

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ReturnOp::getOperationName() { return "quantum.return"; }

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(
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

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::input() { return getODSOperands(0); }

::mlir::MutableOperandRange ReturnOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState) {
  build(odsBuilder, odsState, llvm::None);
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState,
                     ::mlir::ValueRange input) {
  odsState.addOperands(input);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                     ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                     ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verify() {
  if (failed(ReturnOpAdaptor(*this).verify(this->getLoc())))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) &&
            ((v.getType()
                  .cast<::mlir::ShapedType>()
                  .getElementType()
                  .isF64())))) {
        return emitOpError("operand #")
               << index << " must be tensor of 64-bit float values, but got "
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

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser,
                                    ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> inputOperands;
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputTypes;

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperands)) return ::mlir::failure();
  if (!inputOperands.empty()) {
    if (parser.parseColon()) return ::mlir::failure();

    if (parser.parseTypeList(inputTypes)) return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) return ::mlir::failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &p) {
  p << "quantum.return";
  if (!input().empty()) {
    p << ' ';
    p << input();
    p << ' ' << ":";
    p << ' ';
    p << input().getTypes();
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

void ReturnOp::getEffects(
    ::mlir::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
        &effects) {}

}  // namespace quantum
}  // namespace mlir

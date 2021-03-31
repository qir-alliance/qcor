#pragma once
#include "Quantum/QuantumDialect.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

namespace qcor {

class ClangToMLIRTypeVisitor
    : public clang::TypeVisitor<ClangToMLIRTypeVisitor, mlir::Type> {
 public:
  std::string& type_as_str;
  mlir::MLIRContext& context;

  ClangToMLIRTypeVisitor(mlir::MLIRContext& ctx, std::string& type_as_str_arg)
      : type_as_str(type_as_str_arg), context(ctx) {}
  mlir::Type VisitBuiltinType(const clang::BuiltinType* BT) {
    if (BT->isIntegerType()) {
      switch (BT->getKind()) {
        case BuiltinType::Short: {
          return mlir::IntegerType::get(&context, 16);
        } break;
        case BuiltinType::Int: {
          return mlir::IntegerType::get(&context, 32);
        } break;
        case BuiltinType::Long: {
          return mlir::IntegerType::get(&context, 64);
        } break;
        case BuiltinType::LongLong: {
          return mlir::IntegerType::get(&context, 64);
        } break;
        case BuiltinType::ULongLong: {
          return mlir::IntegerType::get(
              &context, 64, mlir::IntegerType::SignednessSemantics::Unsigned);
        } break;
        case BuiltinType::ULong: {
          return mlir::IntegerType::get(
              &context, 64, mlir::IntegerType::SignednessSemantics::Unsigned);
        } break;
        case BuiltinType::UInt: {
          return mlir::IntegerType::get(
              &context, 32, mlir::IntegerType::SignednessSemantics::Unsigned);

        } break;
        case BuiltinType::UShort: {
          return mlir::IntegerType::get(
              &context, 16, mlir::IntegerType::SignednessSemantics::Unsigned);

        } break;
        case BuiltinType::Bool: {
          return mlir::IntegerType::get(&context, 1);
        } break;
        default:
          if (BT->isBooleanType()) {
            return mlir::IntegerType::get(&context, 1);
            break;
          }
          BT->dump();
          llvm_unreachable("unknown integral type");
      };
      // return CIL::IntegerTy::get(kind, qual, &mlirContext);
    } else if (BT->isFloatingType()) {
      // FIXME DO THIS
    }
    return mlir::Type();
  }

  // This will handle qubit = Qubit*
  mlir::Type VisitTypedefType(const clang::TypedefType* r) {
    if (r->isPointerType()) {
      auto qual_type = r->getPointeeType();
      if (qual_type.getAsString().find("Qubit") != std::string::npos) {
        return mlir::OpaqueType::get(&context,
                                     mlir::Identifier::get("quantum", &context),
                                     llvm::StringRef("Qubit"));
      }
    }
    r->dump();
    return mlir::Type();
  }

  // This can handle qcor::qreg&
  mlir::Type VisitLValueReferenceType(const clang::LValueReferenceType* r) {

    auto qual_type = r->getPointeeType();
    if (qual_type.getAsString().find("qreg") != std::string::npos) {
      type_as_str = "qcor::qreg";
      return mlir::OpaqueType::get(&context,
                                   mlir::Identifier::get("quantum", &context),
                                   llvm::StringRef("Array"));
    }
    r->dump();
    return mlir::Type();
  }

  mlir::Type VisitElaboratedType(const clang::ElaboratedType* t) {
    if (t->getNamedType().getAsString().find("vector") != std::string::npos) {
      if (t->getNamedType().getAsString().find("double") != std::string::npos) {
        type_as_str = "std::vector<double>";
      } else if (t->getNamedType().getAsString().find("int") !=
                 std::string::npos) {
        type_as_str = "std::vector<int>";
      }
      return mlir::OpaqueType::get(&context,
                                   mlir::Identifier::get("quantum", &context),
                                   llvm::StringRef("Array"));
    }

    return mlir::Type();
  }
 
};

mlir::Type convertClangType(const clang::Type* type, std::string& type_as_str,
                            mlir::MLIRContext& context) {
  ClangToMLIRTypeVisitor visitor(context, type_as_str);
  return visitor.Visit(type);
}

mlir::Type convertReturnType(const clang::DeclSpec& spec,
                             std::string& ret_type_str,
                             mlir::MLIRContext& context) {
  auto tspectype = spec.getTypeSpecType();

  mlir::Type return_type;
  switch (tspectype) {
    // case DeclSpec::TST_unspecified:
    case DeclSpec::TST_void:
      ret_type_str = "void";
      break;  // do nothing
    // case DeclSpec::TST_char:
    //   return "char";
    // case DeclSpec::TST_wchar:
    //   return Policy.MSWChar ? "__wchar_t" : "wchar_t";
    // case DeclSpec::TST_char8:
    //   return "char8_t";
    // case DeclSpec::TST_char16:
    //   return "char16_t";
    // case DeclSpec::TST_char32:
    //   return "char32_t";
    case DeclSpec::TST_int:
      return_type = mlir::IntegerType::get(&context, 32);
      ret_type_str = "int";
      break;
    // case DeclSpec::TST_int128:
    //   return_type = mlir::IntegerType::get(&context, 128);
    //   ret_type_str = "int";
    // case DeclSpec::TST_extint:
    //   return "_ExtInt";
    // case DeclSpec::TST_half:
    //   return "half";
    case DeclSpec::TST_float:
      return_type = mlir::Float32Type::get(&context);
      ret_type_str = "float";
      break;
    case DeclSpec::TST_double:
      return_type = mlir::Float64Type::get(&context);
      ret_type_str = "double";
      break;
    // case DeclSpec::TST_accum:
    //   return "_Accum";
    // case DeclSpec::TST_fract:
    //   return "_Fract";
    // case DeclSpec::TST_float16:
    //   return_type = mlir::Float16Type::get(&context);

    // case DeclSpec::TST_float128:
    //   return_type = mlir::Float64Type::get(&context);
    case DeclSpec::TST_bool:
      return_type = mlir::IntegerType::get(&context, 1);
      ret_type_str = "bool";
      break;
    case DeclSpec::TST_decimal32:
      return_type = mlir::Float32Type::get(&context);
      ret_type_str = "float";
      break;
    case DeclSpec::TST_decimal64:
      return_type = mlir::Float64Type::get(&context);
      ret_type_str = "double";
      break;
    // case DeclSpec::TST_decimal128:
    //   return_type = mlir::Float64Type::get(&context);
    // case DeclSpec::TST_enum:
    //   return "enum";
    // case DeclSpec::TST_class:
    //   return "class";
    // case DeclSpec::TST_union:
    //   return "union";
    // case DeclSpec::TST_struct:
    //   return "struct";
    // case DeclSpec::TST_interface:
    //   return "__interface";
    // case DeclSpec::TST_typename:
    //   return "type-name";
    // case DeclSpec::TST_typeofType:
    // case DeclSpec::TST_typeofExpr:
    //   return "typeof";
    // case DeclSpec::TST_auto:
    //   return "auto";
    // case DeclSpec::TST_auto_type:
    //   return "__auto_type";
    // case DeclSpec::TST_decltype:
    //   return "(decltype)";
    // case DeclSpec::TST_decltype_auto:
    //   return "decltype(auto)";
    // case DeclSpec::TST_underlyingType:
    //   return "__underlying_type";
    // case DeclSpec::TST_unknown_anytype:
    //   return "__unknown_anytype";
    // case DeclSpec::TST_atomic:
    //   return "_Atomic";
    // case DeclSpec::TST_BFloat16:
    //   return_type = mlir::Float16Type::get(&context);
    // case DeclSpec::TST_error:
    //   return "(error)";
    default:
      if (ret_type_str.empty()) llvm_unreachable("Unknown return typespec!");
  }

  return return_type;
}
}  // namespace qcor
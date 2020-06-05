#pragma once
#include "llvm/IR/LLVMContext.h"
#include <memory>

namespace clang {
    class CodeGenAction;
}

namespace qcor {

std::unique_ptr<clang::CodeGenAction> emit_llvm_ir(const std::string src_code);

}
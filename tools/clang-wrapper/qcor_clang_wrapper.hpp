#pragma once
#include <memory>

#include "llvm/IR/LLVMContext.h"

namespace clang {
class CodeGenAction;
}

namespace qcor {

std::unique_ptr<clang::CodeGenAction> emit_llvm_ir(
    const std::string src_code, std::vector<std::string> extra_headers = {});

}
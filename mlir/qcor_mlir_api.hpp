/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace qcor {

enum OutputType { MLIR, LLVMMLIR, LLVMIR };

const std::string mlir_compile(const std::string &src,
                               const std::string &kernel_name,
                               const OutputType &output_type,
                               bool add_entry_point, int opt_level = 3, std::map<std::string,std::string> extra_args = {});

int execute(const std::string &src, const std::string &kernel_name,
            int opt_level = 3, std::map<std::string, std::string> extra_args = {});

int execute(const std::string &src, const std::string &kernel_name,
            std::vector<std::unique_ptr<llvm::Module>> &extra_code_to_link,
            int opt_level = 3);
} // namespace qcor
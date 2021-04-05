#pragma once

#include <string>
#include <vector>

namespace __internal__ {
namespace qcor {
std::string construct_kernel_subtype(
    std::string src_code, const std::string kernel_name,
    const std::vector<std::string> &program_arg_types,
    const std::vector<std::string> &program_parameters,
    std::vector<std::string> bufferNames);
}
}  // namespace __internal__
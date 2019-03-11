
#include "QCORCompiler.hpp"

namespace qcor {

std::shared_ptr<IR> QCORCompiler::compile(const std::string &src,
                                            std::shared_ptr<Accelerator> acc) {

  return nullptr;
}

std::shared_ptr<IR> QCORCompiler::compile(const std::string &src) {
  return compile(src, nullptr);
}

const std::string
QCORCompiler::translate(const std::string &bufferVariable,
                          std::shared_ptr<Function> function) {
  return "";
}

}

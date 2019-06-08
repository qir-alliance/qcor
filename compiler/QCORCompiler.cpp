
#include "QCORCompiler.hpp"
#include "IRProvider.hpp"
#include "xacc_service.hpp"

namespace qcor {

std::shared_ptr<IR> QCORCompiler::compile(const std::string &src,
                                            std::shared_ptr<Accelerator> acc) {
  return nullptr;
}

std::shared_ptr<IR> QCORCompiler::compile(const std::string &src) {
  return compile(src, nullptr);
}

const std::shared_ptr<Function>
QCORCompiler::compile(std::shared_ptr<Function> f, std::shared_ptr<Accelerator> acc) {

   auto provider = xacc::getService<xacc::IRProvider>("quantum");
   auto ir = provider->createIR();
   ir->addKernel(f);

   // FIXME Hardware Independent Transformation

   // Hardware Dependent Transformations
   if (acc) {
       auto ts = acc->getIRTransformations();
       for (auto& t : ts) {
           ir = t->transform(ir);
       }
   }

   // FIXME Program Verification???

   return f;
}

const std::string
QCORCompiler::translate(const std::string &bufferVariable,
                          std::shared_ptr<Function> function) {
  xacc::error("QCORCompiler::translate() not implemented.");
  return "";
}

}

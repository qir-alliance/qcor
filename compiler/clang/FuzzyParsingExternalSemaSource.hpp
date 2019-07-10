#ifndef COMPILER_FUZZYPARSINGEXTERNALSEMASOURCE_HPP_
#define COMPILER_FUZZYPARSINGEXTERNALSEMASOURCE_HPP_

#include "QCORExternalSemaSource.hpp"

using namespace clang;

namespace qcor {
namespace compiler {
class FuzzyParsingExternalSemaSource : public QCORExternalSemaSource {
private:
  std::vector<std::string> validInstructions;

public:
  FuzzyParsingExternalSemaSource() = default;
  void initialize() override;

  bool LookupUnqualified(clang::LookupResult &R, clang::Scope *S) override;
};
} // namespace compiler
} // namespace qcor
#endif
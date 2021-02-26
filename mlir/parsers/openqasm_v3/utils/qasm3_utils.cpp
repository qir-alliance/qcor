#include "qasm3_utils.hpp"

namespace qcor {

void printErrorMessage(const std::string msg, bool do_exit) {
  std::cout << "\n[OpenQASM3 MLIRGen] Error\n" << msg << "\n\n";
  if (do_exit) exit(1);
}

void printErrorMessage(const std::string msg, mlir::Value v) {
  printErrorMessage(msg, false);
  v.dump();
  exit(1);
}

void printErrorMessage(const std::string msg, std::vector<mlir::Value>&& v) {
  printErrorMessage(msg, false);
  for (auto vv : v) vv.dump();
  exit(1);
}

mlir::Location get_location(mlir::OpBuilder builder,
                            const std::string& file_name,
                            antlr4::ParserRuleContext* context) {
  auto line = context->getStart()->getLine();
  auto col = context->getStart()->getCharPositionInLine();
  return builder.getFileLineColLoc(builder.getIdentifier(file_name), line, col);
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

std::map<std::string, mlir::CmpIPredicate> antlr_to_mlir_predicate{
    {"==", mlir::CmpIPredicate::eq},  {"!=", mlir::CmpIPredicate::ne},
    {"<=", mlir::CmpIPredicate::sle}, {">=", mlir::CmpIPredicate::sge},
    {"<", mlir::CmpIPredicate::slt},  {">", mlir::CmpIPredicate::sgt}};

}  // namespace qcor
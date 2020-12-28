
#include "staq_parser.hpp"

#include "quantum_dialect.hpp"

namespace qasm_parser {

StaqToMLIR::StaqToMLIR(mlir::MLIRContext &context) : builder(&context) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  // This is a generic function, the return type will be inferred later.
  // Arguments type are uniformly unranked tensors.
  llvm::SmallVector<mlir::Type, 4> arg_types;
  auto func_type = builder.getFunctionType(arg_types, llvm::None);
  auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), "main", func_type);
  mlir::FuncOp function(proto);
  auto &entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);
  theModule.push_back(function);

}

void StaqToMLIR::addReturn() {
   builder.create<mlir::quantum::ReturnOp>(builder.getUnknownLoc());
}

void StaqToMLIR::visit(VarAccess &) {}
// Expressions
void StaqToMLIR::visit(BExpr &) {}
void StaqToMLIR::visit(UExpr &) {}
void StaqToMLIR::visit(PiExpr &) {}
void StaqToMLIR::visit(IntExpr &) {}
void StaqToMLIR::visit(RealExpr &r) {}
void StaqToMLIR::visit(VarExpr &v) {}
void StaqToMLIR::visit(ResetStmt &) {}
void StaqToMLIR::visit(IfStmt &) {}
void StaqToMLIR::visit(BarrierGate &) {}
void StaqToMLIR::visit(GateDecl &) {}
void StaqToMLIR::visit(OracleDecl &) {}
void StaqToMLIR::visit(RegisterDecl &d) {
  if (d.is_quantum()) {
    auto size = d.size();
    auto name = d.id();

    auto pos = d.pos();
    auto line = pos.get_linenum();
    auto col = pos.get_column();
    auto fname =
        pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
    auto location =
        builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

    auto integer_type = builder.getI64Type();
    auto integer_attr = mlir::IntegerAttr::get(integer_type, size);
    auto str_attr = builder.getStringAttr(name);
    builder.create<mlir::quantum::QallocOp>(location, integer_attr, str_attr);
  }
}
void StaqToMLIR::visit(AncillaDecl &) {}
void StaqToMLIR::visit(Program &prog) {
  // Program body
  prog.foreach_stmt([this](auto &stmt) { stmt.accept(*this); });
}
void StaqToMLIR::visit(MeasureStmt &m) {
  auto pos = m.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("mz");
  // params
  auto dataType = mlir::VectorType::get({1}, builder.getF64Type());
  std::vector<double> v{0.0};
  auto params_arr_ref = llvm::makeArrayRef(v);
  auto params_dataAttribute =
      mlir::DenseElementsAttr::get(dataType, params_arr_ref);

  // qbits
  auto qbits_dataType = mlir::VectorType::get({1}, builder.getI64Type());
  std::vector<std::int64_t> vv{m.q_arg().offset().value()};
  auto qbits_arr_ref = llvm::makeArrayRef(vv);
  auto qbits_dataAttribute =
      mlir::DenseIntElementsAttr::get(qbits_dataType, qbits_arr_ref);

  auto qreg_dataType = mlir::VectorType::get({1}, builder.getI64Type());
  std::vector<llvm::StringRef> vvv{m.q_arg().var()};
  auto qreg_arr_ref = llvm::makeArrayRef(vvv);
  auto qreg_dataAttribute =
      mlir::DenseStringElementsAttr::get(qreg_dataType, qreg_arr_ref);
  builder.create<mlir::quantum::InstOp>(location, str_attr, qreg_dataAttribute,
                                        qbits_dataAttribute,
                                        params_dataAttribute);
}
void StaqToMLIR::visit(UGate &u) {
  auto pos = u.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("u3");
  // params
  auto dataType = mlir::VectorType::get({3}, builder.getF64Type());
  std::vector<double> v{u.theta().constant_eval().value(),
                        u.phi().constant_eval().value(),
                        u.lambda().constant_eval().value()};
  auto params_arr_ref = llvm::makeArrayRef(v);
  auto params_dataAttribute =
      mlir::DenseElementsAttr::get(dataType, params_arr_ref);

  // qbits
  auto qbits_dataType = mlir::VectorType::get({1}, builder.getI64Type());
  std::vector<std::int64_t> vv{u.arg().offset().value()};
  auto qbits_arr_ref = llvm::makeArrayRef(vv);
  auto qbits_dataAttribute =
      mlir::DenseIntElementsAttr::get(qbits_dataType, qbits_arr_ref);

  // qreg
  auto qreg_dataType = mlir::VectorType::get({1}, builder.getI64Type());
  std::vector<llvm::StringRef> vvv{u.arg().var()};
  auto qreg_arr_ref = llvm::makeArrayRef(vvv);
  auto qreg_dataAttribute =
      mlir::DenseStringElementsAttr::get(qreg_dataType, qreg_arr_ref);
  builder.create<mlir::quantum::InstOp>(location, str_attr, qreg_dataAttribute,
                                        qbits_dataAttribute,
                                        params_dataAttribute);
}
void StaqToMLIR::visit(CNOTGate &g) {
  auto pos = g.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("cx");

  // params
  auto dataType = mlir::VectorType::get({1}, builder.getF64Type());
  std::vector<double> v{0.0};
  auto params_arr_ref = llvm::makeArrayRef(v);
  auto params_dataAttribute =
      mlir::DenseElementsAttr::get(dataType, params_arr_ref);

  // qbits
  auto qbits_dataType = mlir::VectorType::get({2}, builder.getI64Type());
  std::vector<std::int64_t> vv{g.ctrl().offset().value(),
                               g.tgt().offset().value()};
  auto qbits_arr_ref = llvm::makeArrayRef(vv);
  auto qbits_dataAttribute =
      mlir::DenseIntElementsAttr::get(qbits_dataType, qbits_arr_ref);

  // qreg
  auto qreg_dataType = mlir::VectorType::get({2}, builder.getI64Type());
  std::vector<llvm::StringRef> vvv{g.ctrl().var(), g.tgt().var()};
  auto qreg_arr_ref = llvm::makeArrayRef(vvv);
  auto qreg_dataAttribute =
      mlir::DenseStringElementsAttr::get(qreg_dataType, qreg_arr_ref);

  builder.create<mlir::quantum::InstOp>(location, str_attr, qreg_dataAttribute,
                                        qbits_dataAttribute,
                                        params_dataAttribute);
}
//   void visit(BarrierGate&) = 0;
void StaqToMLIR::visit(DeclaredGate &g) {
  auto pos = g.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr(g.name());

  // params
  auto dataType = mlir::VectorType::get(
      {g.num_cargs() == 0 ? 1 : g.num_cargs()}, builder.getF64Type());
  std::vector<double> v;  //{0.0};
  for (int i = 0; i < g.num_cargs(); i++) {
    v.push_back(g.carg(i).constant_eval().value());
  }
  if (v.empty()) {
    v.push_back(0.0);
  }
  auto params_arr_ref = llvm::makeArrayRef(v);
  auto params_dataAttribute =
      mlir::DenseElementsAttr::get(dataType, params_arr_ref);

  // qbits
  auto qbits_dataType =
      mlir::VectorType::get({g.num_qargs()}, builder.getI64Type());
  std::vector<std::int64_t> vv;
  for (int i = 0; i < g.num_qargs(); i++) {
    vv.push_back(g.qarg(i).offset().value());
  }
  auto qbits_arr_ref = llvm::makeArrayRef(vv);
  auto qbits_dataAttribute =
      mlir::DenseIntElementsAttr::get(qbits_dataType, qbits_arr_ref);

  // qreg
  auto qreg_dataType =
      mlir::VectorType::get({g.num_qargs()}, builder.getI64Type());
  std::vector<llvm::StringRef> vvv;
  for (int i = 0; i < g.num_qargs(); i++) {
    vvv.push_back(g.qarg(i).var());
  }
  auto qreg_arr_ref = llvm::makeArrayRef(vvv);
  auto qreg_dataAttribute =
      mlir::DenseStringElementsAttr::get(qreg_dataType, qreg_arr_ref);
  builder.create<mlir::quantum::InstOp>(location, str_attr, qreg_dataAttribute,
                                        qbits_dataAttribute,
                                        params_dataAttribute);
}

}  // namespace qasm_parser
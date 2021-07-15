#include "openqasm_mlir_generator.hpp"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace qcor {

std::set<std::string_view> default_inline_overrides{
    "x",  "y",  "z",  "h",  "s",  "sdg",  "t", "tdg",
    "rx", "ry", "rz", "cz", "cy", "swap", "cx"};

static std::vector<std::string> builtins{
    "u3", "u2",   "u1",  "cx",  "id",  "u0",  "x",   "y",  "z",
    "h",  "s",    "sdg", "t",   "tdg", "rx",  "ry",  "rz", "cz",
    "cy", "swap", "ch",  "ccx", "crz", "cu1", "cu2", "cu3"};

static std::vector<std::string> search_for_inliner{
    "u3", "u2",  "u1", "cx",  "id", "u0", "x",  "y",  "z",  "h",
    "s",  "sdg", "t",  "tdg", "rx", "ry", "rz", "cz", "cy", "swap"};

static std::map<std::string, std::string> missing_builtins{
    {"u", R"#(
gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; })#"},
    {"p", R"#(gate p(theta) a
{
  rz(theta) a;
})#"},
    {"sx", R"#(gate sx a { sdg a; h a; sdg a; })#"},
    {"sxdg", R"#(gate sxdg a { s a; h a; s a; })#"},
    {"cswap", R"#(gate cswap a,b,c
{
  cx c,b;
  ccx a,b,c;
  cx c,b;
})#"},
    {"crx", R"#(gate crx(lambda) a,b
{
  u1(pi/2) b;
  cx a,b;
  u3(-lambda/2,0,0) b;
  cx a,b;
  u3(lambda/2,-pi/2,0) b;
})#"},
    {"cry", R"#(gate cry(lambda) a,b
{
  u3(lambda/2,0,0) b;
  cx a,b;
  u3(-lambda/2,0,0) b;
  cx a,b;
  ry(lambda) a;
})#"},
    {"cp", R"#(gate cp(lambda) a,b
{
  p(lambda/2) a;
  cx a,b;
  p(-lambda/2) b;
  cx a,b;
  p(lambda/2) b;
})#"},
    {"csx", R"#(gate csx a,b { h b; cu1(pi/2) a,b; h b; })#"},
    {"cu", R"#(gate cu(theta,phi,lambda,gamma) c, t
{ p(gamma) c;
  p((lambda+phi)/2) c;
  p((lambda-phi)/2) t;
  cx c,t;
  u(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u(theta/2,phi,0) t;
})#"},
    {"rxx", R"#(gate rxx(theta) a,b
{
  u3(pi/2, theta, 0) a;
  h b;
  cx a,b;
  u1(-theta) b;
  cx a,b;
  h b;
  u2(-pi, pi-theta) a;
})#"},
    {"rzz", R"#(gate rzz(theta) a,b
{
  cx a,b;
  u1(theta) b;
  cx a,b;
})#"},
    {"rccx", R"#(gate rccx a,b,c
{
  u2(0,pi) c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  cx a, c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  u2(0,pi) c;
})#"}};

template <class Op>
void split(const std::string &s, char delim, Op op) {
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, delim);) {
    *op++ = item;
  }
}

inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

void CountGateDecls::visit(GateDecl &g) {
  auto name = g.id();
  if (std::find(builtins.begin(), builtins.end(), name) == builtins.end()) {
    gates_to_inline.push_back(name);
  }
  count++;
}

class MapParameterSubExpr : public staq::ast::Visitor {
 protected:
  mlir::OpBuilder &builder;
  mlir::Location &location;
  std::map<std::string, mlir::Value> &symbol_table;

  mlir::Value current_value;

 public:
  MapParameterSubExpr(mlir::OpBuilder &b, mlir::Location &l,
                      std::map<std::string, mlir::Value> &symbols)
      : builder(b), location(l), symbol_table(symbols) {}
  mlir::Value getValue() { return current_value; }
  void visit(VarAccess &) override {}
  void visit(BExpr &expr) override {
    Expr &left = expr.lexp();
    Expr &right = expr.rexp();
    mlir::Value left_value, right_value;
    if (left.constant_eval().has_value()) {
      auto constant_value = left.constant_eval().value();
      left_value = builder.create<mlir::ConstantOp>(
          location, mlir::FloatAttr::get(builder.getF64Type(), constant_value));
    } else {
      left.accept(*this);
      left_value = current_value;
    }

    if (right.constant_eval().has_value()) {
      auto constant_value = right.constant_eval().value();
      right_value = builder.create<mlir::ConstantOp>(
          location, mlir::FloatAttr::get(builder.getF64Type(), constant_value));
    } else {
      right.accept(*this);
      right_value = current_value;
    }

    if (expr.op() == BinaryOp::Divide) {
      current_value =
          builder.create<mlir::DivFOp>(location, left_value, right_value);
    } else if (expr.op() == BinaryOp::Plus) {
      current_value =
          builder.create<mlir::AddFOp>(location, left_value, right_value);
    } else if (expr.op() == BinaryOp::Minus) {
      current_value =
          builder.create<mlir::SubFOp>(location, left_value, right_value);
    } else if (expr.op() == BinaryOp::Times) {
      current_value =
          builder.create<mlir::MulFOp>(location, left_value, right_value);
    } else if (expr.op() == BinaryOp::Pow) {
      std::cout << "[OpenQASM MLIR Gen] pow(x,y) not supported yet.\n";
      exit(1);
    }
  }

  void visit(UExpr &expr) override {
    Expr &sub = expr.subexp();
    sub.accept(*this);
    if (expr.op() == UnaryOp::Neg) {
      current_value = builder.create<mlir::NegFOp>(location, current_value);
    } else {
      std::cout << "[OpenQASM MLIR Gen] no other unary ops supported.\n";
      exit(1);
    }
  }

  void visit(PiExpr &) override {}
  void visit(IntExpr &) override {}
  void visit(RealExpr &r) override {}
  void visit(VarExpr &v) override {
    if (symbol_table.count(v.var())) {
      current_value = symbol_table[v.var()];
    } else {
      std::cout << "[OpenQasm MLIR Gen] Error, " << v.var()
                << " is not a valid var in the symbol table.\n";
    }
  }
  void visit(ResetStmt &) override {}
  void visit(IfStmt &) override {}
  void visit(BarrierGate &) override {}
  void visit(GateDecl &) override {}
  void visit(OracleDecl &) override {}
  void visit(RegisterDecl &) override {}
  void visit(AncillaDecl &) override {}
  void visit(Program &prog) override {}
  void visit(MeasureStmt &m) override {}
  void visit(UGate &u) override {}
  void visit(CNOTGate &cx) override {}
  void visit(DeclaredGate &g) override {}
};

void OpenQasmMLIRGenerator::visit(Program &prog) {
  // How many statements are there (starts with 25)
  auto n_stmts = prog.body().size();
  // How many gatedecls are there?
  std::size_t n_gate_decls = 0;
  CountGateDecls count_gate_decls(n_gate_decls);
  prog.foreach_stmt([&](auto &stmt) { stmt.accept(count_gate_decls); });

  for (auto &g : count_gate_decls.gates_to_inline) {
    default_inline_overrides.insert(g);
  }

  // INLINE any complex controlled gates from stdlib
  staq::transformations::Inliner::config config;
  config.overrides = default_inline_overrides;
  staq::transformations::inline_ast(prog);

  // If n_stmts > n_gate_decls, then we need a main function
  add_main = (n_stmts > n_gate_decls);
  m_module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Useful opaque type defs
  llvm::StringRef qubit_type_name("Qubit"), array_type_name("Array"),
      result_type_name("Result");
  mlir::Identifier dialect = mlir::Identifier::get("quantum", &context);
  qubit_type = mlir::OpaqueType::get(&context, dialect, qubit_type_name);
  array_type = mlir::OpaqueType::get(&context, dialect, array_type_name);
  result_type = mlir::OpaqueType::get(&context, dialect, result_type_name);
  auto int_type = builder.getI32Type();
  auto argv_type =
      mlir::OpaqueType::get(&context, dialect, llvm::StringRef("ArgvType"));
  auto qreg_type =
      mlir::OpaqueType::get(&context, dialect, llvm::StringRef("qreg"));

  if (add_main) {
    std::vector<mlir::Type> arg_types_vec2{};
    auto func_type2 =
        builder.getFunctionType(llvm::makeArrayRef(arg_types_vec2), llvm::None);
    auto proto2 = mlir::FuncOp::create(
        builder.getUnknownLoc(), "__internal_mlir_" + file_name, func_type2);
    mlir::FuncOp function2(proto2);
    auto save_main_entry_block = function2.addEntryBlock();

    if (add_entry_point) {
      std::vector<mlir::Type> arg_types_vec{int_type, argv_type};
      auto func_type =
          builder.getFunctionType(llvm::makeArrayRef(arg_types_vec), int_type);
      auto proto =
          mlir::FuncOp::create(builder.getUnknownLoc(), "main", func_type);
      mlir::FuncOp function(proto);
      main_entry_block = function.addEntryBlock();
      auto &entryBlock = *main_entry_block;
      builder.setInsertionPointToStart(&entryBlock);

      auto main_args = main_entry_block->getArguments();
      llvm::ArrayRef<mlir::Attribute> tmp{};
      builder.create<mlir::quantum::QRTInitOp>(
          builder.getUnknownLoc(), main_args[0], main_args[1],
          mlir::ArrayAttr::get(tmp, builder.getContext()));

      // call the function from main, run finalize, and return 0
      builder.create<mlir::CallOp>(builder.getUnknownLoc(), function2);
      builder.create<mlir::quantum::QRTFinalizeOp>(builder.getUnknownLoc());
      is_first_inst = false;
      auto integer_attr = mlir::IntegerAttr::get(builder.getI32Type(), 0);
      mlir::Value ret_zero = builder.create<mlir::ConstantOp>(
          builder.getUnknownLoc(), integer_attr);
      builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), ret_zero);
      m_module.push_back(function);
      function_names.push_back("main");
    }

    std::vector<mlir::Type> arg_types_vec3{qreg_type};
    auto func_type3 =
        builder.getFunctionType(llvm::makeArrayRef(arg_types_vec3), llvm::None);
    auto proto3 =
        mlir::FuncOp::create(builder.getUnknownLoc(), file_name, func_type3);
    mlir::FuncOp function3(proto3);

    auto tmp = function3.addEntryBlock();
    builder.setInsertionPointToStart(tmp);
    builder.create<mlir::quantum::SetQregOp>(builder.getUnknownLoc(),
                                             tmp->getArguments()[0]);
    builder.create<mlir::CallOp>(builder.getUnknownLoc(), function2);
    builder.create<mlir::quantum::QRTFinalizeOp>(builder.getUnknownLoc());
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                   llvm::ArrayRef<mlir::Value>());
    builder.setInsertionPointToStart(save_main_entry_block);

    m_module.push_back(function2);
    m_module.push_back(function3);
    function_names.push_back("__internal_mlir_" + file_name);
    function_names.push_back(file_name);

    main_entry_block = save_main_entry_block;

    // Create function_name(opaque.qreg q), setup main to call it
  }
  prog.foreach_stmt([this](auto &stmt) { stmt.accept(*this); });
}

void OpenQasmMLIRGenerator::initialize_mlirgen(
    bool _add_entry_point, const std::string function,
    std::map<std::string, std::string> extra_quantum_args) {
  file_name = function;
  add_entry_point = _add_entry_point;
}

void OpenQasmMLIRGenerator::mlirgen(const std::string &src) {
  using namespace staq;

  std::string src_copy = src;

  // Make sure we have the preamble text
  std::string preamble = "OPENQASM 2.0;";
  auto preamble_start = src.find(preamble);
  if (preamble_start == std::string::npos) {
    std::cout << "[OpenQASM MLIR Gen] Error, no OPENQASM 2.0 preamble text.\n";
    exit(1);
  }

  preamble = "include \"qelib1.inc\";";
  preamble_start = src.find(preamble);
  if (preamble_start == std::string::npos) {
    std::cout << "[OpenQASM MLIR Gen] Error, no include \"qelib1.inc\" "
                 "preamble text.\n";
    exit(1);
  }

  // Add any required missing pre-defines that we 
  // know the impl for.
  std::vector<std::string> added;
  std::string extra_insts = "\n";
  auto lines = split(src, '\n');
  for (auto line : lines) {
    if (line.find("OPENQASM") == std::string::npos &&
        line.find("include") == std::string::npos &&
        line.find("measure") == std::string::npos &&
        line.find("qreg") == std::string::npos &&
        line.find("creg") == std::string::npos && !line.empty()) {
      auto inst_name = split(line, ' ')[0];
      if (inst_name.find("(") != std::string::npos) {
        inst_name = inst_name.substr(0, inst_name.find("("));
      }
      if (std::find(builtins.begin(), builtins.end(), inst_name) ==
              builtins.end() &&
          std::find(added.begin(), added.end(), inst_name) == added.end()) {
        extra_insts += missing_builtins[inst_name] + "\n";
        added.push_back(inst_name);
      }
    }
  }
  src_copy.insert(preamble_start+preamble.length(), extra_insts);

  ast::ptr<ast::Program> prog;
  try {
    prog = parser::parse_string(src_copy);
    // transformations::inline_ast(*prog);
    transformations::desugar(*prog);
  } catch (std::exception &e) {
    std::stringstream ss;
    std::cout << e.what() << "\n";
  }

  visit(*prog);

  return;
}

void OpenQasmMLIRGenerator::finalize_mlirgen() {
  // FIXME Need to deallocate any allocated qalloc results.
  for (auto &[qreg_name, qalloc_op] : qubit_allocations) {
    builder.create<mlir::quantum::DeallocOp>(builder.getUnknownLoc(),
                                             qalloc_op);
  }

  if (add_main) {
    // builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), llvm::None);
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                   llvm::ArrayRef<mlir::Value>());
  }
}

void OpenQasmMLIRGenerator::visit(GateDecl &gate_function) {
  auto name = gate_function.id();

  if (std::find(builtins.begin(), builtins.end(), name) == builtins.end()) {
    std::vector<mlir::Type> arg_types;

    function_names.push_back(name);

    auto cn_args = gate_function.c_params().size();
    for (std::size_t i = 0; i < cn_args; i++) {
      arg_types.push_back(builder.getF64Type());
    }

    auto n_args = gate_function.q_params().size();
    for (std::size_t i = 0; i < n_args; i++) {
      arg_types.push_back(qubit_type);
    }

    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), name, func_type);
    mlir::FuncOp function(proto);
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    auto arguments = entryBlock.getArguments();

    for (std::size_t i = 0; i < cn_args; i++) {
      auto argument = arguments[i];
      auto arg_name = gate_function.c_params()[i];
      temporary_sub_kernel_args.insert({arg_name, argument});
    }

    for (std::size_t i = 0; i < n_args; i++) {
      auto argument = arguments[cn_args + i];
      auto arg_name = gate_function.q_params()[i];
      temporary_sub_kernel_args.insert({arg_name, argument});
    }
    in_sub_kernel = true;

    gate_function.foreach_stmt([this](Gate &g) { g.accept(*this); });

    in_sub_kernel = false;
    temporary_sub_kernel_args.clear();
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    m_module.push_back(function);

    if (add_main) builder.setInsertionPointToStart(main_entry_block);
  }
}

void OpenQasmMLIRGenerator::visit(RegisterDecl &d) {
  if (d.is_quantum()) {
    std::int64_t size = d.size();
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
    auto allocation = builder.create<mlir::quantum::QallocOp>(
        location, array_type, integer_attr, str_attr);
    qubit_allocations.insert({name, allocation});
  }
}

void OpenQasmMLIRGenerator::visit(MeasureStmt &m) {
  auto pos = m.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("mz");

  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_var_name = m.q_arg().var();
  if (!qubit_allocations.count(qreg_var_name)) {
    // throw an error
  }

  std::uint64_t qidx = m.q_arg().offset().value();
  auto qbit_key = std::make_pair(qreg_var_name, qidx);
  mlir::Value qbit_value;
  if (extracted_qubits.count(qbit_key)) {
    qbit_value = extracted_qubits[qbit_key];
  } else {
    auto qubits = qubit_allocations[qreg_var_name].qubits();

    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, qubit_type, qubits, pos);

    extracted_qubits.insert({std::make_pair(qreg_var_name, qidx), qbit_value});
  }
  qubits_for_inst.push_back(qbit_value);

  builder.create<mlir::quantum::InstOp>(
      location, result_type, str_attr, llvm::makeArrayRef(qubits_for_inst),
      llvm::makeArrayRef(std::vector<mlir::Value>{}));
}

void OpenQasmMLIRGenerator::visit(UGate &u) {
  auto pos = u.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("u3");

  std::vector<mlir::Value> params_for_inst;
  auto &theta_expr = u.theta();

  MapParameterSubExpr visitor(builder, location, temporary_sub_kernel_args);
  if (theta_expr.constant_eval().has_value()) {
    double val = theta_expr.constant_eval().value();
    auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), val);
    mlir::Value val_val =
        builder.create<mlir::ConstantOp>(location, float_attr);
    params_for_inst.push_back(val_val);
  } else {
    theta_expr.accept(visitor);
    params_for_inst.push_back(visitor.getValue());
  }

  auto &phi_expr = u.phi();
  if (phi_expr.constant_eval().has_value()) {
    double val = phi_expr.constant_eval().value();
    auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), val);
    mlir::Value val_val =
        builder.create<mlir::ConstantOp>(location, float_attr);
    params_for_inst.push_back(val_val);
  } else {
    phi_expr.accept(visitor);
    params_for_inst.push_back(visitor.getValue());
  }

  auto &lambda_expr = u.lambda();
  if (lambda_expr.constant_eval().has_value()) {
    double val = lambda_expr.constant_eval().value();
    auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), val);
    mlir::Value val_val =
        builder.create<mlir::ConstantOp>(location, float_attr);
    params_for_inst.push_back(val_val);
  } else {
    lambda_expr.accept(visitor);
    params_for_inst.push_back(visitor.getValue());
  }

  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_var_name = u.arg().var();
  if (!qubit_allocations.count(qreg_var_name)) {
    // throw an error
  }

  mlir::Value qbit_value;
  if (u.arg().offset().has_value()) {
    std::uint64_t qidx = u.arg().offset().value();
    auto qbit_key = std::make_pair(qreg_var_name, qidx);
    if (extracted_qubits.count(qbit_key)) {
      qbit_value = extracted_qubits[qbit_key];
    } else {
      auto qubits = qubit_allocations[qreg_var_name].qubits();

      auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
      mlir::Value pos =
          builder.create<mlir::ConstantOp>(location, integer_attr);
      qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
          location, qubit_type, qubits, pos);

      extracted_qubits.insert(
          {std::make_pair(qreg_var_name, qidx), qbit_value});
    }
  } else {
    auto var_name = u.arg().var();
    qbit_value = temporary_sub_kernel_args[var_name];
  }
  qubits_for_inst.push_back(qbit_value);

  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qubits_for_inst), llvm::makeArrayRef(params_for_inst));
}

void OpenQasmMLIRGenerator::visit(CNOTGate &g) {
  auto pos = g.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  // if (is_first_inst && !in_sub_kernel) {
  //   auto main_args = main_entry_block->getArguments();

  //   builder.create<mlir::quantum::QRTInitOp>(location, main_args[0],
  //                                            main_args[1]);
  //   is_first_inst = false;
  // }
  auto str_attr = builder.getStringAttr("cx");

  // params
  mlir::DenseElementsAttr params_dataAttribute;

  // ctrl qbits
  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_ctrl_var_name = g.ctrl().var();
  auto qreg_tgt_var_name = g.tgt().var();

  if (!qubit_allocations.count(qreg_ctrl_var_name)) {
    // throw an error
  }

  if (!qubit_allocations.count(qreg_tgt_var_name)) {
    // throw an error
  }

  // Get CTRL Qubit MLIR Value
  std::uint64_t ctrl_idx = g.ctrl().offset().value();
  auto ctrlqkey = std::make_pair(qreg_ctrl_var_name, ctrl_idx);

  mlir::Value ctl_qbit_value;
  if (extracted_qubits.count(ctrlqkey)) {
    ctl_qbit_value = extracted_qubits[ctrlqkey];
  } else {
    auto qubits = qubit_allocations[qreg_ctrl_var_name].qubits();

    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), ctrl_idx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    ctl_qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, qubit_type, qubits, pos);

    extracted_qubits.insert(
        {std::make_pair(qreg_ctrl_var_name, ctrl_idx), ctl_qbit_value});
  }
  qubits_for_inst.push_back(ctl_qbit_value);

  // Get Target Qubit MLIR Value
  std::uint64_t tgt_idx = g.tgt().offset().value();
  auto tgtqkey = std::make_pair(qreg_tgt_var_name, tgt_idx);

  mlir::Value tgt_qbit_value;
  if (extracted_qubits.count(tgtqkey)) {
    tgt_qbit_value = extracted_qubits[tgtqkey];
  } else {
    auto qubits = qubit_allocations[qreg_tgt_var_name].qubits();

    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), tgt_idx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    tgt_qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, qubit_type, qubits, pos);

    extracted_qubits.insert(
        {std::make_pair(qreg_tgt_var_name, tgt_idx), tgt_qbit_value});
  }
  qubits_for_inst.push_back(tgt_qbit_value);

  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qubits_for_inst),
      llvm::makeArrayRef(std::vector<mlir::Value>{}));
}

void OpenQasmMLIRGenerator::visit(DeclaredGate &g) {
  auto pos = g.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr(g.name());

  // params
  std::vector<mlir::Value> params_for_inst;
  if (g.num_cargs()) {
    for (int i = 0; i < g.num_cargs(); i++) {
      if (g.carg(i).constant_eval().has_value()) {
        double val = g.carg(i).constant_eval().value();

        auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), val);
        mlir::Value val_val =
            builder.create<mlir::ConstantOp>(location, float_attr);
        params_for_inst.push_back(val_val);
      } else {
        MapParameterSubExpr visitor(builder, location,
                                    temporary_sub_kernel_args);
        Expr &theta_expr = g.carg(i);
        theta_expr.accept(visitor);
        params_for_inst.push_back(visitor.getValue());
      }
    }
  }

  // qbits
  std::vector<mlir::Value> qubits_for_inst;
  for (int i = 0; i < g.num_qargs(); i++) {
    auto qreg_var_name = g.qarg(i).var();
    if (!qubit_allocations.count(qreg_var_name)) {
      // throw an error
    }

    if (!in_sub_kernel) {
      std::uint64_t qidx = g.qarg(i).offset().value();
      auto qbit_key = std::make_pair(qreg_var_name, qidx);
      mlir::Value qbit_value;
      if (extracted_qubits.count(qbit_key)) {
        qbit_value = extracted_qubits[qbit_key];
      } else {
        auto qubits = qubit_allocations[qreg_var_name].qubits();

        auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
        mlir::Value pos =
            builder.create<mlir::ConstantOp>(location, integer_attr);
        qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
            location, qubit_type, qubits, pos);

        extracted_qubits.insert(
            {std::make_pair(qreg_var_name, qidx), qbit_value});
      }
      qubits_for_inst.push_back(qbit_value);
    } else {
      auto qubit_kernel_arg = temporary_sub_kernel_args[qreg_var_name];
      qubits_for_inst.push_back(qubit_kernel_arg);
    }
  }

  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qubits_for_inst), llvm::makeArrayRef(params_for_inst));
}

}  // namespace qcor
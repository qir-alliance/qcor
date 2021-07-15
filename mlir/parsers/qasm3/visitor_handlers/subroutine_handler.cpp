#include "qasm3_visitor.hpp"

namespace {
// Helper to generate a QIR callable wrapper for a QASM3 subroutine:
// A Callable is constructed from a functor table (array of size 4)
// for body (base), adjoint, controlled, and controlled adjoint functors
// that all have signature of void(Tuple, Tuple, Tuple).
// This method generates those 4 wrappers as well as the function to construct
// the Callable.
void add_callable_gen(mlir::OpBuilder &builder, const std::string &func_name,
                      mlir::ModuleOp &moduleOp, mlir::FuncOp &wrapped_func) {
  auto context = builder.getContext();
  auto main_block = builder.saveInsertionPoint();
  mlir::Identifier dialect = mlir::Identifier::get("quantum", context);
  llvm::StringRef tuple_type_name("Tuple");
  auto tuple_type = mlir::OpaqueType::get(context, dialect, tuple_type_name);
  llvm::StringRef array_type_name("Array");
  auto array_type = mlir::OpaqueType::get(context, dialect, array_type_name);
  llvm::StringRef callable_type_name("Callable");
  auto callable_type =
      mlir::OpaqueType::get(context, dialect, callable_type_name);
  llvm::StringRef qubit_type_name("Qubit");
  auto qubit_type = mlir::OpaqueType::get(context, dialect, qubit_type_name);

  const std::vector<mlir::Type> argument_types{tuple_type, tuple_type,
                                               tuple_type};
  auto func_type = builder.getFunctionType(argument_types, llvm::None);
  const std::string BODY_WRAPPER_SUFFIX = "__body__wrapper";
  const std::string ADJOINT_WRAPPER_SUFFIX = "__adj__wrapper";
  const std::string CTRL_WRAPPER_SUFFIX = "__ctl__wrapper";
  const std::string CTRL_ADJOINT_WRAPPER_SUFFIX = "__ctladj__wrapper";

  std::vector<mlir::FuncOp> all_wrapper_funcs;
  {
    // Body wrapper:
    const std::string wrapper_fn_name = func_name + BODY_WRAPPER_SUFFIX;
    mlir::FuncOp function_op(mlir::FuncOp::create(builder.getUnknownLoc(),
                                                  wrapper_fn_name, func_type));
    function_op.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto &entryBlock = *function_op.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    auto arguments = entryBlock.getArguments();
    assert(arguments.size() == 3);
    mlir::Value arg_tuple = arguments[1];
    auto fn_type = wrapped_func.getType().cast<mlir::FunctionType>();
    mlir::TypeRange arg_types(fn_type.getInputs());
    auto unpackOp = builder.create<mlir::quantum::TupleUnpackOp>(
        builder.getUnknownLoc(), arg_types, arg_tuple);
    auto call_op = builder.create<mlir::CallOp>(
        builder.getUnknownLoc(), wrapped_func, unpackOp.result());
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    moduleOp.push_back(function_op);
    all_wrapper_funcs.emplace_back(function_op);
  }

  {
    // Adjoint wrapper:
    const std::string wrapper_fn_name = func_name + ADJOINT_WRAPPER_SUFFIX;
    mlir::FuncOp function_op(mlir::FuncOp::create(builder.getUnknownLoc(),
                                                  wrapper_fn_name, func_type));
    function_op.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto &entryBlock = *function_op.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    // Wrap the call to the body wrapperin StartAdjointURegion and
    // EndAdjointURegion
    builder.create<mlir::quantum::StartAdjointURegion>(builder.getUnknownLoc());
    mlir::FuncOp body_wrapper = all_wrapper_funcs[0];
    // Forward tuple arguments to the body (will unpack there)
    auto call_op = builder.create<mlir::CallOp>(
        builder.getUnknownLoc(), body_wrapper, entryBlock.getArguments());
    builder.create<mlir::quantum::EndAdjointURegion>(builder.getUnknownLoc());
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    moduleOp.push_back(function_op);
    all_wrapper_funcs.emplace_back(function_op);
  }
  {
    // Controlled wrapper:
    const std::string wrapper_fn_name = func_name + CTRL_WRAPPER_SUFFIX;
    mlir::FuncOp function_op(mlir::FuncOp::create(builder.getUnknownLoc(),
                                                  wrapper_fn_name, func_type));
    function_op.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto &entryBlock = *function_op.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    auto arguments = entryBlock.getArguments();
    assert(arguments.size() == 3);
    mlir::Value arg_tuple = arguments[1];
    auto fn_type = wrapped_func.getType().cast<mlir::FunctionType>();
    // Unpack to Array + Tuple (Array = controlled bits)
    // { Array + { Body Tuple } }
    // FIXME: currently, we can only handle single-qubit control
    // TODO: update EndCtrlURegion to take an array of qubits.
    mlir::TypeRange arg_types({array_type, tuple_type});
    auto unpackOp = builder.create<mlir::quantum::TupleUnpackOp>(
        builder.getUnknownLoc(), arg_types, arg_tuple);
    mlir::FuncOp body_wrapper = all_wrapper_funcs[0];
    mlir::Value control_array = unpackOp.result()[0];
    mlir::Value body_arg_tuple = unpackOp.result()[1];

    // Extract the control qubit:
    mlir::Value qubit_idx = builder.create<mlir::ConstantOp>(
        builder.getUnknownLoc(),
        mlir::IntegerAttr::get(builder.getI64Type(), 0));
    mlir::Value ctrl_qubit = builder.create<mlir::quantum::ExtractQubitOp>(
        builder.getUnknownLoc(), qubit_type, control_array, qubit_idx);

    // Call the body wrapped in StartCtrlURegion/EndCtrlURegion
    builder.create<mlir::quantum::StartCtrlURegion>(builder.getUnknownLoc());
    auto call_op = builder.create<mlir::CallOp>(
        builder.getUnknownLoc(), body_wrapper,
        llvm::ArrayRef<mlir::Value>(
            {arguments[0], body_arg_tuple, arguments[2]}));
    builder.create<mlir::quantum::EndCtrlURegion>(builder.getUnknownLoc(),
                                                  ctrl_qubit);
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    moduleOp.push_back(function_op);
    all_wrapper_funcs.emplace_back(function_op);
  }
  {
    // Controlled Adjoint wrapper:
    const std::string wrapper_fn_name = func_name + CTRL_ADJOINT_WRAPPER_SUFFIX;
    mlir::FuncOp function_op(mlir::FuncOp::create(builder.getUnknownLoc(),
                                                  wrapper_fn_name, func_type));
    function_op.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto &entryBlock = *function_op.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    // Wrap the call to the ctrl wrapper wrapped in StartAdjointURegion and
    // EndAdjointURegion
    builder.create<mlir::quantum::StartAdjointURegion>(builder.getUnknownLoc());
    mlir::FuncOp ctrl_wrapper = all_wrapper_funcs[2];
    // Forward tuple arguments to the controlled (will unpack there)
    auto call_op = builder.create<mlir::CallOp>(
        builder.getUnknownLoc(), ctrl_wrapper, entryBlock.getArguments());
    builder.create<mlir::quantum::EndAdjointURegion>(builder.getUnknownLoc());
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    moduleOp.push_back(function_op);
    all_wrapper_funcs.emplace_back(function_op);
  }

  // Add a function to create the callable wrapper for this kernel
  auto create_callable_func_type = builder.getFunctionType({}, callable_type);
  const std::string create_callable_fn_name = func_name + "__callable";
  auto create_callable_func_proto =
      mlir::FuncOp::create(builder.getUnknownLoc(), create_callable_fn_name,
                           create_callable_func_type);
  mlir::FuncOp create_callable_function_op(create_callable_func_proto);
  auto &create_callable_entryBlock =
      *create_callable_function_op.addEntryBlock();
  builder.setInsertionPointToStart(&create_callable_entryBlock);
  auto callable_create_op = builder.create<mlir::quantum::CreateCallableOp>(
      builder.getUnknownLoc(), callable_type,
      builder.getSymbolRefAttr(wrapped_func));
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                 callable_create_op.callable());
  moduleOp.push_back(create_callable_function_op);
  builder.restoreInsertionPoint(main_block);
}
}; // namespace
namespace qcor {
antlrcpp::Any qasm3_visitor::visitSubroutineDefinition(
    qasm3Parser::SubroutineDefinitionContext *context) {
  // : 'def' Identifier ( LPAREN classicalArgumentList? RPAREN )?
  //   quantumArgumentList? returnSignature? subroutineBlock

  // subroutineBlock
  // : LBRACE statement* returnStatement? RBRACE
  subroutine_return_statment_added = false;
  auto subroutine_name = context->Identifier()->getText();
  std::vector<mlir::Type> argument_types;
  std::vector<std::string> arg_names;
  std::vector<std::vector<std::string>> arg_attributes;
  if (auto arg_list = context->classicalArgumentList()) {
    // list of type:varname
    for (auto arg : arg_list->classicalArgument()) {
      auto type = arg->classicalType()->getText();
      argument_types.push_back(
          convertQasm3Type(arg->classicalType(), symbol_table, builder));
      arg_names.push_back(arg->association()->Identifier()->getText());
      arg_attributes.push_back({});
    }
  }

  if (context->quantumArgumentList()) {
    for (auto quantum_arg : context->quantumArgumentList()->quantumArgument()) {
      if (quantum_arg->quantumType()->getText() == "qubit" &&
          !quantum_arg->designator()) {
        arg_attributes.push_back({});
        argument_types.push_back(qubit_type);
      } else {
        argument_types.push_back(array_type);
        auto designator = quantum_arg->designator()->getText();
        if (designator == "[DYNAMIC]") {
          arg_attributes.push_back({""});
        } else {
          auto qreg_size =
              symbol_table.evaluate_constant_integer_expression(designator);
          arg_attributes.push_back({std::to_string(qreg_size)});
        }
      }

      arg_names.push_back(quantum_arg->association()->Identifier()->getText());
    }
  }

  bool has_return = false;
  mlir::Type return_type = builder.getIntegerType(32);
  if (context->returnSignature()) {
    has_return = true;
    auto classical_type = context->returnSignature()->classicalType();
    // can return bit, bit[], uint[], int[], float[], bool
    if (classical_type->bitType()) {
      if (auto designator = classical_type->designator()) {
        auto bit_size = symbol_table.evaluate_constant_integer_expression(
            designator->getText());
        llvm::ArrayRef<int64_t> shape(bit_size);
        return_type = mlir::MemRefType::get(shape, builder.getI1Type());
      } else {
        return_type = builder.getI1Type();
      }
    } else if (auto single_desig = classical_type->singleDesignatorType()) {
      if (single_desig->getText() == "float") {
        auto bit_width = symbol_table.evaluate_constant_integer_expression(
            classical_type->designator()->getText());
        if (bit_width == 16) {
          return_type = builder.getF16Type();
        } else if (bit_width == 32) {
          return_type = builder.getF32Type();
        } else if (bit_width == 64) {
          return_type = builder.getF64Type();
        } else {
          printErrorMessage(
              "on subroutine return type - we only support 16, 32, or 64 width "
              "floating point types.",
              context);
        }
      } else if (single_desig->getText() == "int") {
        auto bit_width = symbol_table.evaluate_constant_integer_expression(
            classical_type->designator()->getText());
        return_type = builder.getIntegerType(bit_width);
      } else {
        printErrorMessage(
            "we do not yet support this subroutine single designator return "
            "type.",
            context);
      }
    } else if (auto no_desig = classical_type->noDesignatorType()) {
      if (no_desig->getText().find("uint") != std::string::npos) {
        return_type = builder.getIntegerType(32, false);
      } else if (no_desig->getText().find("int") != std::string::npos) {
        return_type = builder.getIntegerType(32);
      } else if (no_desig->getText().find("float") != std::string::npos) {
        return_type = builder.getF32Type();
      } else if (no_desig->getText().find("double") != std::string::npos) {
        return_type = builder.getF64Type();
      } else if (no_desig->getText().find("int64_t") != std::string::npos) {
        return_type = builder.getI64Type();
      } else {
        printErrorMessage("Invalid no-designator default type.", context);
      }
    } else {
      printErrorMessage("Alex implement other return types.", context);
    }
  }

  current_function_return_type = return_type;

  auto main_block = builder.saveInsertionPoint();

  mlir::FuncOp function, interop_function;
  if (has_return) {
    auto func_type = builder.getFunctionType(argument_types, return_type);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), subroutine_name,
                                      func_type);
    mlir::FuncOp function2(proto);
    function = function2;
  } else {
    auto func_type = builder.getFunctionType(argument_types, llvm::None);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), subroutine_name,
                                      func_type);
    mlir::FuncOp function2(proto);
    function = function2;
  }
  // Handle "extern" subroutine declaration:
  if (context->subroutineBlock()->EXTERN()) {
    // std::cout << "Handle extern subroutine: " << subroutine_name << "\n";
    builder.setInsertionPointToStart(&m_module.getRegion().getBlocks().front());
    auto func_decl = builder.create<mlir::FuncOp>(
        get_location(builder, file_name, context), subroutine_name,
        function.getType().cast<mlir::FunctionType>());
    func_decl.setVisibility(mlir::SymbolTable::Visibility::Private);
    builder.restoreInsertionPoint(main_block);
    symbol_table.add_seen_function(subroutine_name, function);
    return 0;
  }

  auto &entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  symbol_table.enter_new_scope();

  auto arguments = entryBlock.getArguments();
  for (int i = 0; i < arguments.size(); i++) {
    symbol_table.add_symbol(arg_names[i], arguments[i], arg_attributes[i]);
  }

  auto quantum_block = context->subroutineBlock();

  auto ret = visitChildren(quantum_block);

  if (!subroutine_return_statment_added) {
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
  }
  m_module.push_back(function);

  builder.restoreInsertionPoint(main_block);

  symbol_table.exit_scope();

  symbol_table.add_seen_function(subroutine_name, function);

  subroutine_return_statment_added = false;

  symbol_table.set_last_created_block(nullptr);

  // Add __interop__ function here so we can invoke from C++.
  mlir::FunctionType interop_func_type;
  // QASM3 subroutines have classical args followed by qubit args
  // for qcor interop, we need qubit/qreg arg first.
  // FIXME Only do this for subroutines with a single qubit[] array.
  std::reverse(argument_types.begin(), argument_types.end());
  if (has_return) {
    interop_func_type = builder.getFunctionType(argument_types, return_type);
  } else {
    interop_func_type = builder.getFunctionType(argument_types, llvm::None);
  }

  auto interop =
      mlir::FuncOp::create(builder.getUnknownLoc(),
                           subroutine_name + "__interop__", interop_func_type);
  auto &interop_entryBlock = *interop.addEntryBlock();
  builder.setInsertionPointToStart(&interop_entryBlock);

  std::vector<mlir::BlockArgument> vec_to_reverse;
  for (int i = 0; i < arguments.size(); i++) {
    vec_to_reverse.push_back(interop_entryBlock.getArgument(i));
  }
  std::reverse(vec_to_reverse.begin(), vec_to_reverse.end());

  auto call_op_interop = builder.create<mlir::CallOp>(
      builder.getUnknownLoc(), function, llvm::makeArrayRef(vec_to_reverse));
  if (has_return) {
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                   call_op_interop.getResults());
  } else {
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
  }

  builder.restoreInsertionPoint(main_block);

  m_module.push_back(interop);

  // TODO: add a compile switch to enable/disable this export:
  add_callable_gen(builder, subroutine_name, m_module, function);
  return 0;
}

antlrcpp::Any qasm3_visitor::visitReturnStatement(
    qasm3Parser::ReturnStatementContext *context) {
  is_return_stmt = true;
  auto location = get_location(builder, file_name, context);

  auto ret_stmt = context->statement()->getText();
  ret_stmt.erase(std::remove(ret_stmt.begin(), ret_stmt.end(), ';'),
                 ret_stmt.end());

  mlir::Value value;
  if (symbol_table.has_symbol(ret_stmt)) {
    value = symbol_table.get_symbol(ret_stmt);
    // Actually return value if it is a bit[],
    // load and return if it is a bit

    if (current_function_return_type) { // this means it is a subroutine
      if (!current_function_return_type.isa<mlir::MemRefType>()) {
        if (current_function_return_type.isa<mlir::IntegerType>() &&
            current_function_return_type.getIntOrFloatBitWidth() == 1) {
          // This is a bit and not a bit[]
          auto tmp = get_or_create_constant_index_value(0, location, 64,
                                                        symbol_table, builder);
          llvm::ArrayRef<mlir::Value> zero_index(tmp);
          if (value.getType().isa<mlir::MemRefType>() &&
              value.getType().cast<mlir::MemRefType>().getShape().empty()) {
            // No need to add index if the Memref has no dimension.
            value = builder.create<mlir::LoadOp>(location, value);
          } else {
            value = builder.create<mlir::LoadOp>(location, value, zero_index);
          }
        } else {
          value =
              builder.create<mlir::LoadOp>(location, value); //, zero_index);
        }
      } else {
        printErrorMessage("We do not return memrefs from subroutines.",
                          context);
      }
    } else {
      if (auto t = value.getType().dyn_cast_or_null<mlir::MemRefType>()) {
        if (t.getRank() == 0) {
          value = builder.create<mlir::LoadOp>(location, value);
        }
      }
    }

  } else {
    if (auto expr_stmt = context->statement()->expressionStatement()) {
      qasm3_expression_generator exp_generator(builder, symbol_table,
                                               file_name);
      exp_generator.visit(expr_stmt);
      value = exp_generator.current_value;
    } else {
      visitChildren(context->statement());
      value = symbol_table.get_last_value_added();
    }
  }
  is_return_stmt = false;

  builder.create<mlir::ReturnOp>(
      builder.getUnknownLoc(),
      llvm::makeArrayRef(std::vector<mlir::Value>{value}));
  subroutine_return_statment_added = true;
  return 0;
}
} // namespace qcor
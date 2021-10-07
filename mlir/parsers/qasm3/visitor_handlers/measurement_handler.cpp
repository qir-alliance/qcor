/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include <numeric>

#include "qasm3_visitor.hpp"

namespace qcor {

//     quantumMeasurement
//     : 'measure' indexIdentifierList
//     ;

// quantumMeasurementAssignment
//     : quantumMeasurement ( ARROW indexIdentifierList)?
//     | indexIdentifierList EQUALS quantumMeasurement
//     ;

antlrcpp::Any qasm3_visitor::visitQuantumMeasurement(
    qasm3Parser::QuantumMeasurementContext* context) {
  printErrorMessage(
      "visiting this node is not implemented, havent seen it yet. if you hit "
      "this, let alex know",
      context);
  return 0;
}

antlrcpp::Any qasm3_visitor::visitQuantumMeasurementAssignment(
    qasm3Parser::QuantumMeasurementAssignmentContext* context) {
  auto location = get_location(builder, file_name, context);
  auto str_attr = builder.getStringAttr("mz");

  if (is_return_stmt) {
    // should be something like measure q;
    auto qubit_or_qreg = context->quantumMeasurement()
                             ->indexIdentifierList()
                             ->indexIdentifier(0)
                             ->Identifier();
    auto measured_object = symbol_table.get_symbol(qubit_or_qreg->getText());
    if (measured_object.getType() == qubit_type) {
      mlir::Value instop =
          builder
              .create<mlir::quantum::InstOp>(
                  location, result_type, str_attr,
                  llvm::makeArrayRef(std::vector<mlir::Value>{measured_object}),
                  llvm::makeArrayRef(std::vector<mlir::Value>{}))
              .bit();
      symbol_table.add_symbol("return_value_" + qubit_or_qreg->getText(),
                              instop);
      return 0;
    } else {
      // array type
      // create memref of size qreg_size, measure all qubits,
      // and store the bit results to each element of the memref,
      // then return that.
      // NOT IMPLEMENTED YET
      printErrorMessage("visiting this node is not implemented, havent seen it "
                        "yet. if you hit "
                        "this, let alex know",
                        context);

      return 0;
    }
  }

  // Handle all other measure stmts
  if (context->EQUALS() || context->ARROW()) {
    // we have EXPR = measure EXPR
    auto indexIdentifierList = context->indexIdentifierList();
    auto measured_list = context->quantumMeasurement()->indexIdentifierList();

    // Get the Qubits
    std::string measured_qreg =
        measured_list->indexIdentifier(0)->Identifier()->getText();
    auto value = symbol_table.get_symbol(measured_qreg);

    // Get the Bits
    auto bit_variable_name =
        indexIdentifierList->indexIdentifier(0)->Identifier()->getText();
    auto bit_value = symbol_table.get_symbol(bit_variable_name);

    // First handle the case measure qubit, or measure qubit[i];
    if (value.getType() == array_type &&
        measured_list->indexIdentifier(0)->expressionList()) {
      // Here we are measuring a qubit from an array

      // let's get that single qubit value
      auto idx_str = measured_list->indexIdentifier(0)
                         ->expressionList()
                         ->expression(0)
                         ->getText();

      try {
        value = get_or_extract_qubit(measured_qreg, std::stoi(idx_str),
                                     location, symbol_table, builder);
      } catch (...) {
        if (symbol_table.has_symbol(idx_str)) {
          auto qubits = symbol_table.get_symbol(measured_qreg);
          auto qbit = symbol_table.get_symbol(idx_str);
          llvm::StringRef qubit_type_name("Qubit");
          mlir::Identifier dialect =
              mlir::Identifier::get("quantum", builder.getContext());
          auto qubit_type = mlir::OpaqueType::get(builder.getContext(), dialect,
                                                  qubit_type_name);
          if (!qbit.getType().isa<mlir::IntegerType>()) {
            qbit = builder.create<mlir::IndexCastOp>(
                location, builder.getI64Type(), qbit);
          }
          value = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qubits, qbit);
        } else {
          printErrorMessage(
              "Invalid measurement index on the given qubit register: " +
                  measured_qreg + ", " + idx_str,
              context);
        }
      }
    } else if (value.getType() == array_type &&
               measured_list->indexIdentifier(0)->rangeDefinition()) {
      // This is measuring a range...

      if (auto bit_range =
              indexIdentifierList->indexIdentifier(0)->rangeDefinition()) {
        if (bit_range->expression().size() > 2) {
          printErrorMessage(
              "we only support measuring ranges with [start:stop]");
        }

        // Get bit and qubit ranges integers...
        auto ba = symbol_table.evaluate_constant_integer_expression(
            bit_range->expression(0)->getText());
        auto bb = symbol_table.evaluate_constant_integer_expression(
            bit_range->expression(1)->getText());
        auto qa = symbol_table.evaluate_constant_integer_expression(
            measured_list->indexIdentifier(0)
                ->rangeDefinition()
                ->expression(0)
                ->getText());
        auto qb = symbol_table.evaluate_constant_integer_expression(
            measured_list->indexIdentifier(0)
                ->rangeDefinition()
                ->expression(1)
                ->getText());

        auto nqubits_in_register =
            value.getDefiningOp<mlir::quantum::QallocOp>()
                .size()
                .getLimitedValue();
        if (qb >= nqubits_in_register) {
          printErrorMessage("End value for qubit range is >= n_qubits ( = " +
                                std::to_string(nqubits_in_register) +
                                ") in register.\n",
                            context);
        }

        auto n_bits_in_register =
            bit_value.getType().cast<mlir::MemRefType>().getShape()[0];
        if (bb >= n_bits_in_register) {
          printErrorMessage("End value for bit range is >= n_bits ( = " +
                                std::to_string(n_bits_in_register) +
                                ") in register.\n",
                            context);
        }

        if ((bb - ba) != (qb - qa)) {
          printErrorMessage("Range sizes must be equal.", context);
        }

        auto size = bb - ba;
        std::vector<int> qubit_indices(size), bit_indices(size);
        std::iota(std::begin(qubit_indices), std::end(qubit_indices), qa);
        std::iota(std::begin(bit_indices), std::end(bit_indices), ba);
        qubit_indices.push_back(qb);
        bit_indices.push_back(bb);

        for (int i = 0; i <= size; i++) {
          auto qbit_idx = qubit_indices[i];
          auto bit_idx = bit_indices[i];

          // !IMPORTANT! q.extract expects i64 as index.
          // Using index type will cause validation issue at the MLIR level.
          // (i.e. requires all-the-way-to-LLVM lowering for types to match)
          mlir::Value idx_val = get_or_create_constant_integer_value(
              qbit_idx, location, builder.getI64Type(), symbol_table, builder);
          mlir::Value bit_idx_val = get_or_create_constant_index_value(
              bit_idx, location, 64, symbol_table, builder);

          auto extract_qubit = get_or_extract_qubit(measured_qreg, qbit_idx, location,
                                              symbol_table, builder);

          auto instop = builder.create<mlir::quantum::InstOp>(
              location, result_type, str_attr,
              llvm::makeArrayRef(extract_qubit),
              llvm::makeArrayRef(std::vector<mlir::Value>{}));
          auto cast_bit_op = builder.create<mlir::quantum::ResultCastOp>(
              location, builder.getIntegerType(1), instop.bit());
          builder.create<mlir::StoreOp>(
              location, cast_bit_op.bit_result(), bit_value,
              llvm::makeArrayRef(std::vector<mlir::Value>{bit_idx_val}));
          symbol_table.invalidate_qubit_extracts(measured_qreg, {qbit_idx});
        }

        return 0;

      } else {
        printErrorMessage(
            "If you are measuring a qubit range you must store to a bit "
            "range.",
            context);
      }
    }

    // Ok, now if this is a qubit type, and not an array
    if (value.getType() == qubit_type) {
      auto instop = builder.create<mlir::quantum::InstOp>(
          location, result_type, str_attr, llvm::makeArrayRef(value),
          llvm::makeArrayRef(std::vector<mlir::Value>{}));
      const std::string qubit_var_name =
          symbol_table.get_symbol_var_name(value);
      if (!qubit_var_name.empty() && qubit_var_name != measured_qreg) {
        symbol_table.erase_symbol(qubit_var_name);
      }

      // Get the bit or bit[]
      mlir::Value v;
      if (auto index_list =
              indexIdentifierList->indexIdentifier(0)->expressionList()) {
        // Need to extract element from bit array to set it
        qasm3_expression_generator equals_exp_generator(builder, symbol_table,
                                                        file_name);
        equals_exp_generator.visit(index_list->expression(0));
        v = equals_exp_generator.current_value;
        // Make sure v is of Index type (to be used w/ StoreOp)
        v = builder.create<mlir::IndexCastOp>(location, builder.getIndexType(),
                                              v);
      } else {
        v = get_or_create_constant_index_value(
          0, location, 64, symbol_table, builder);
      }

      assert(v.getType().isa<mlir::IndexType>());

      // Cast Measure Result -> Bit (i1)
      auto cast_bit_op = builder.create<mlir::quantum::ResultCastOp>(
          location, builder.getIntegerType(1), instop.bit());

      if (bit_value.getType().isa<mlir::MemRefType>() &&
          bit_value.getType().cast<mlir::MemRefType>().getShape().empty()) {
        if (enable_nisq_ifelse) {
          // Track the Result* associated with the bit in the Symbol Table
          symbol_table.add_measure_bit_assignment(bit_value, instop.bit());
        }
        // If the array is a **zero-dimemsion** Memref *without* shape
        // we don't send on the index (probably v = 0).
        // This will fail to validate at the MLIR level (Memref dimension mismatches)
        // (at LLVM level, it doesn't matter b/w a memref<i1> vs. memref<1xi1>).
        builder.create<mlir::StoreOp>(location, cast_bit_op.bit_result(),
                                      bit_value);
      } else {
        if (enable_nisq_ifelse) {
          if (!symbol_table.has_symbol(indexIdentifierList->getText())) {
            // Added a measure Result* tracking to the bit array element:
            // e.g. track var name 'c[1]' -> Result*
            symbol_table.add_symbol(indexIdentifierList->getText(),
                                    cast_bit_op.bit_result());
          }
          symbol_table.add_measure_bit_assignment(cast_bit_op.bit_result(),
                                                  instop.bit());
        }
        builder.create<mlir::StoreOp>(
            location, cast_bit_op.bit_result(), bit_value,
            llvm::makeArrayRef(std::vector<mlir::Value>{v}));
      }
    } else {
      // This is the case where we are measuring an entire qubit array
      // to a bit array
      // First check that the sizes match up
      std::uint64_t nqubits = 0;
      if (auto qalloc_op = value.getDefiningOp<mlir::quantum::QallocOp>()) {
        nqubits = qalloc_op.size().getLimitedValue();
      } else {
        // then this is a block arg, we should have a size attribute
        auto attributes = symbol_table.get_variable_attributes(measured_qreg);
        if (!attributes.empty()) {
          try {
            nqubits = std::stoi(attributes[0]);
          } catch (...) {
            printErrorMessage(
                "Could not infer qubit[] size from block argument.", context);
          }
        } else {
          printErrorMessage(
              "Could not infer qubit[] size from block argument. No size "
              "attribute for variable in symbol table.",
              context);
        }
      }
      auto nbits = bit_value.getType().cast<mlir::MemRefType>().getShape()[0];
      if (nbits != nqubits) {
        printErrorMessage(
            "cannot measure the qubit array to the specified bit array, sizes "
            "do not match: n_qubits = " +
                std::to_string(nqubits) + ", n_bits = " + std::to_string(nbits),
            context);
      }

      const auto qreg_name = symbol_table.get_symbol_var_name(value);
      for (int i = 0; i < nqubits; i++) {
        // q.Extract must use integer type (not index type)
        mlir::Value q_idx_val = get_or_create_constant_integer_value(
            i, location, builder.getI64Type(), symbol_table, builder);
        mlir::Value idx_val = get_or_create_constant_index_value(
            i, location, 64, symbol_table, builder);

        assert(!qreg_name.empty());
        auto extract_qubit =
            get_or_extract_qubit(qreg_name, i, location, symbol_table, builder);

        auto instop = builder.create<mlir::quantum::InstOp>(
            location, result_type, str_attr, llvm::makeArrayRef(extract_qubit),
            llvm::makeArrayRef(std::vector<mlir::Value>{}));
        // Cast Measure Result -> Bit (i1)
        auto cast_bit_op = builder.create<mlir::quantum::ResultCastOp>(
            location, builder.getIntegerType(1), instop.bit());

        builder.create<mlir::StoreOp>(
            location, cast_bit_op.bit_result(), bit_value,
            llvm::makeArrayRef(std::vector<mlir::Value>{idx_val}));
      }
      symbol_table.invalidate_qubit_extracts(qreg_name);
    }

  } else {
    printErrorMessage("invalid measure syntax.", context);
  }
  return 0;
}
}  // namespace qcor
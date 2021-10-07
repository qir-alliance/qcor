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
#include "qir-qrt.hpp"
#include <cstring>
#include <iostream>
extern "C" {
void __quantum__rt__string_update_reference_count(QirString *str,
                                                  int32_t count) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  if (str) {
    str->m_refCount += count;
    if (str->m_refCount <= 0) {
      // Dealloc:
      delete str;
    }
  }
}

QirString *__quantum__rt__string_create(char *null_terminated_buffer) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString(null_terminated_buffer);
}

QirString *__quantum__rt__string_concatenate(QirString *in_head,
                                             QirString *in_tail) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString(in_head->m_str + in_tail->m_str);
}

bool __quantum__rt__string_equal(QirString *lhs, QirString *rhs) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return (lhs->m_str == rhs->m_str);
}

QirString *__quantum__rt__int_to_string(int64_t val) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString(std::to_string(val));
}

QirString *__quantum__rt__double_to_string(double val) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString(std::to_string(val));
}
QirString *__quantum__rt__bool_to_string(bool val) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString(std::to_string(val));
}
QirString *__quantum__rt__result_to_string(Result *val) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString(std::to_string(*val));
}

QirString *__quantum__rt__pauli_to_string(Pauli pauli) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString("Pauli");
}

QirString *__quantum__rt__qubit_to_string(Qubit *q) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString("q[" + std::to_string(q->id) + "]");
}

QirString *__quantum__rt__range_to_string(int64_t range_start,
                                          int64_t range_step,
                                          int64_t range_end) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return new QirString("Range(" + std::to_string(range_start) + "," +
                       std::to_string(range_step) + "," +
                       std::to_string(range_end) + ")");
}

const char *__quantum__rt__string_get_data(QirString *str) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return str->m_str.c_str();
}

int32_t __quantum__rt__string_get_length(QirString *str) {

  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return str->m_str.size();
}
}
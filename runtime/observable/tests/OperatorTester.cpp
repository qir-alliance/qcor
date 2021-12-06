/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "gtest/gtest.h"

#include "qcor_observable.hpp"

TEST(OperatorTester, checkSimple) {

    qcor::Operator op("pauli", "X0 X1"), op2("pauli", "Y0 Y1");

    EXPECT_EQ(2, op.nBits());
    EXPECT_EQ(2, op2.nBits());

    std::cout << op2.toString() << "\n";

    auto t = op + 2.2 * op2;

    std::cout << t.toString() << "\n";

    std::cout << op2.toString() << "\n";

    auto terms = t.getNonIdentitySubTerms();
    EXPECT_EQ(2, terms.size());
    for (auto tt : terms) {
        std::cout << tt << "\n";
    }
    EXPECT_NEAR(2.2, terms[1].coefficient().real(), 1e-3);

    auto el = t.to_sparse_matrix();
    for (auto e : el ) {
        std::cout << e.row() << ", " << e.col() << ", " << e.coeff() << "\n";
    }

    using namespace qcor;
    auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);
    std::cout << H.toString() << "\n";
    std::cout << SP(3).toString() << "\n";

    std::cout << (adag(1)*a(0)).toString() << "\n";

    // FIXME Make this more robust.

}

#include "xacc.hpp"

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  xacc::set_verbose(true);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}

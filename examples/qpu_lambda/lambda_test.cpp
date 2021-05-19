#include "qcor.hpp"

int main(int argc, char** argv) {
  set_shots(1024);
  int n = argc;
  double m = 22;

  auto a = qpu_lambda([](qreg q) {
      print("n was captured, and is ", n);
      print("m was captured, and is ", m);
      for (int i = 0; i < n; i++) {
        H(q[0]);
      }
      Measure(q[0]);
      m++;
  }, n, m);

  // By-value capture
  auto b = qpu_lambda([=](qreg q) {
      print("[By-val] n was captured, and is ", n);
      print("[By-val] m was captured, and is ", m);
      for (int i = 0; i < n; i++) {
        H(q[0]);
      }
      Measure(q[0]);
  }, n, m);

  auto q = qalloc(1);
  a(q);
  auto qb = qalloc(1);
  b(qb);
  print("m after lambda", m);
  // m was modified by the lambda...
  qcor_expect(m == 23);
  
  q.print();
  // Run 1 Hadamard
  qcor_expect(q.counts().size() == 2);
  qcor_expect(q.counts()["0"] > 400);
  qcor_expect(q.counts()["1"] > 400);
  qcor_expect(q.counts()["0"] + q.counts()["1"] == 1024);

  n = 2;
  m = 33.0;
  auto r = qalloc(1);
  auto rb = qalloc(1);
  print("running again to show capture variables are captured by reference and by value");
  a(r);
  b(rb);
  r.print();
  // H - H == I
  qcor_expect(r.counts()["0"] == 1024);
  // b still uses n = 1 (capture by value...)
  rb.print();
  qcor_expect(rb.counts().size() == 2);
  qcor_expect(rb.counts()["0"] > 400);
  qcor_expect(rb.counts()["1"] > 400);
  qcor_expect(rb.counts()["0"] + rb.counts()["1"] == 1024);

  // Test passing an r-val to lambda
  auto ansatz_X0X1 = qpu_lambda([](qreg q, double x) {
    print("ansatz: x = ", x);
    X(q[0]);
    Ry(q[1], x);
    CX(q[1], q[0]);
    H(q);
    Measure(q);
  });

  auto qtest = qalloc(2);
  // Pass an rval...
  ansatz_X0X1(qtest, 1.2334);
  auto exp = qtest.exp_val_z();
  print("<X0X1> = ", exp);

  // Test a loop:
  const std::vector<double> expectedResults{
      0.0,       -0.324699, -0.614213, -0.837166, -0.9694,
      -0.996584, -0.915773, -0.735724, -0.475947, -0.164595,
      0.164595,  0.475947,  0.735724,  0.915773,  0.996584,
      0.9694,    0.837166,  0.614213,  0.324699,  0.0};

  const auto angles = linspace(-M_PI, M_PI, 20);
  for (size_t i = 0; i < angles.size(); ++i) {
    auto buffer = qalloc(2);
    ansatz_X0X1(buffer, angles[i]);
    auto exp = buffer.exp_val_z();
    print("<X0X1>(", angles[i], ") = ", exp, "; expected:", expectedResults[i]);
    qcor_expect(std::abs(expectedResults[i] - exp) < 0.1);
  }

  // Test by-ref argument...
  auto add_one = qpu_lambda([](qreg q, int &result) {
    print("add_one: result =", result);
    result++;
  });
  
  // capture add_one lambda and use by-ref arguments.
  auto add_two = qpu_lambda(
      [](qreg q, int &result) {
        add_one(q, result);
        add_one(q, result);
      },
      add_one);
  auto buffer_test = qalloc(2);
  int test_val = 1;

  add_one(buffer_test, test_val);
  qcor_expect(test_val == 2);

  add_two(buffer_test, test_val);
  qcor_expect(test_val == 4);

  auto add_one_copy = qpu_lambda([](qreg q, int result) {
    print("add_one: entry result =", result);
    result++;
    print("add_one: exit result =", result);
  });

  auto test_val_const = 12;
  add_one_copy(buffer_test, test_val_const);
  // Should stay the same
  qcor_expect(test_val_const == 12);

  auto count_qubits = qpu_lambda([](qreg q, int &result) {
    result = q.size();
  });

  int nb_qubits = 0;
  count_qubits(qalloc(20), nb_qubits);
  std::cout << "Count = " << nb_qubits << "\n";
  qcor_expect(nb_qubits == 20);

  auto vector_sum =
      qpu_lambda([](qreg q, std::vector<double> input, double &result) {
        result = 0.0;
        for (auto &val : input) {
          result = result + val;
        }
      });

  double check = 0.0;
  std::vector<double> vec_to_check { 1.0, 2.0, 3.0 };
  vector_sum(qalloc(1), vec_to_check, check);
  std::cout << "Sum: " << check << "\n";
  qcor_expect(std::abs(check - 6.0) < 1e-12);
  check = 0.0;
  // Inline construction
  vector_sum(qalloc(1), std::vector<double>{2.0, 4.0, 6.0}, check);
  std::cout << "Sum: " << check << "\n";
  qcor_expect(std::abs(check - 12.0) < 1e-12);
}

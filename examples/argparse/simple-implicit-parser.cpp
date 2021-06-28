__qpu__ void f(qreg q) {
  H(q);
  Measure(q);
}

int main(int argc, char** argv) {
    using namespace qcor::arg;

    add_argument("n-qubits")
        .default_value(1)
        .action([](const std::string& value) { return std::stoi(value); });
    parse_args(argc, argv);
    auto N = get_argument<int>("n-qubits");

    auto q = qalloc(N);
    f(q);
    q.print();
}
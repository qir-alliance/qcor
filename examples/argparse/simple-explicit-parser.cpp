__qpu__ void f(qreg q) {
  H(q);
  Measure(q);
}

int main(int argc, char** argv) {
    using namespace qcor::arg;
    
    ArgumentParser program(argv[0]);

    program.add_argument("n-qubits")
            .default_value(1)
            .action([](const std::string& value) { return std::stoi(value); });
    program.parse_args(argc, argv);
    auto N = program.get<int>("n-qubits");

    auto q = qalloc(N);
    f(q);
    q.print();
}
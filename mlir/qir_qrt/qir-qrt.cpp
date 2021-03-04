#include "qir-qrt.hpp"

#include <alloca.h>

// #include "qcor.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"
#include "qcor_config.hpp"
#include "xacc_config.hpp"
#include "config_file_parser.hpp"

Result ResultZeroVal = 0;
Result ResultOneVal = 1;
Result *ResultZero = &ResultZeroVal;
Result *ResultOne = &ResultOneVal;
unsigned long allocated_qbits = 0;
std::shared_ptr<xacc::AcceleratorBuffer> qbits;
std::shared_ptr<xacc::Accelerator> qpu;
std::string qpu_name = "qpp";
std::string qpu_config = "";
enum QRT_MODE { FTQC, NISQ };
QRT_MODE mode;
std::vector<std::unique_ptr<Array>> allocated_arrays;
int shots = 0;
bool verbose = true;
bool external_qreg_provided = false;

bool initialized = false;

void print_help() {
  std::cout << "QCOR QIR Runtime Help Menu\n\n";
  std::cout << "optional arguments:\n";
  std::cout << "  -qpu QPUNAME[:BACKEND] | example -qpu ibm:ibmq_vigo, -qpu aer:ibmq_vigo\n";
  std::cout << "  -qpu-config config_file.ini | example: -qpu ibm:ibmq_vigo -qpu-config ibm_config.ini\n";
  std::cout << "  -qrt QRT_MODE (can be nisq or ftqc) | example -qrt nisq\n";
  std::cout << "  -shots NUMSHOTS (number of shots to use in nisq run)\n";
  std::cout << "  -opt LEVEL | example -opt 1\n";
  std::cout << "  -print-opt-stats (turn on printout of optimization statistics) \n";
  std::cout << "  -v,-verbose,--verbose (run with printouts)\n";
  std::cout << "  -xacc-verbose (turn on extra xacc verbose print-out)\n\n";
  exit(0);
}

void __quantum__rt__initialize(int argc, int8_t** argv) {
  char** casted = reinterpret_cast<char**>(argv);
  std::vector<std::string> args(casted, casted + argc);

  mode = QRT_MODE::FTQC;
  for (int i = 0; i < args.size(); i++) {
    auto arg = args[i];
    if (arg == "-qpu") {
      qpu_name = args[i + 1];
    } else if (arg == "-qpu-config") {
      qpu_config = args[i+1];
    } else if (arg == "-qrt") {
      mode = args[i + 1] == "nisq" ? QRT_MODE::NISQ : QRT_MODE::FTQC;
    } else if (arg == "-shots") {
      shots = std::stoi(args[i + 1]);
    } else if (arg == "-xacc-verbose") {
      verbose = true;
      xacc::set_verbose(true);
    } else if (arg == "-v") {
      verbose = true;
    } else if (arg == "-verbose") {
      verbose = true;
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help") {
      print_help();
    } else if (arg == "-h") {
      print_help();
    } else if (arg == "-opt") {
      xacc::internal_compiler::__opt_level =
        std::stoi(args[i+1]);
    } else if (arg == "-print-opt-stats") {
      xacc::internal_compiler::__print_opt_stats = true;
    }
  }

  initialize();
}

void initialize() {
  if (!initialized) {
    if (verbose) printf("[qir-qrt] Initializing FTQC runtime...\n");
    // qcor::set_verbose(true);
    xacc::internal_compiler::__qrt_env = "ftqc";
    
    // if XACC_INSTALL_DIR != XACC_ROOT
    // then we need to pass --xacc-root-path XACC_ROOT
    //
    // Example - we are on Rigetti QCS and can't install via sudo
    // so we dpkg -x xacc to a user directory, but deb package
    // expects to be extracted to /usr/local/xacc, and xacc_config.hpp
    // points to that /usr/local/xacc. Therefore ServiceRegistry fails
    // to load plugins and libs, unless we change rootPath.
    std::string xacc_config_install_dir = std::string(XACC_INSTALL_DIR);
    std::string qcor_config_xacc_root = std::string(XACC_ROOT);
    if (xacc_config_install_dir != qcor_config_xacc_root) {
      std::vector<std::string> cmd_line{"--xacc-root-path",
                                      qcor_config_xacc_root};
      xacc::Initialize(cmd_line);
    } else {
      xacc::Initialize();
    }

    if (verbose)
      std::cout << "[qir-qrt] Running on " << qpu_name << " backend.\n";
    std::shared_ptr<xacc::Accelerator> qpu;

    xacc::HeterogeneousMap qpu_config_map;
    if (!qpu_config.empty()) {
        auto parser = xacc::getService<xacc::ConfigFileParsingUtil>("ini");
        qpu_config_map = parser->parse(qpu_config);
    }

    if (!qpu_config_map.keyExists<int>("shots") && shots > 0 && mode == QRT_MODE::NISQ) {
      if (verbose) printf("Automatically setting shots for nisq mode execution to %d\n", shots);
      qpu_config_map.insert("shots", shots);
    }
    
    if (mode == QRT_MODE::NISQ) {
      xacc::internal_compiler::__qrt_env = "nisq";
      qpu = xacc::getAccelerator(qpu_name, qpu_config_map);
    } else {
      qpu = xacc::getAccelerator(qpu_name);
    }

    xacc::internal_compiler::qpu = qpu;
    ::quantum::qrt_impl = xacc::getService<::quantum::QuantumRuntime>(
        xacc::internal_compiler::__qrt_env);
    ::quantum::qrt_impl->initialize("empty");
    initialized = true;
  }
}

void __quantum__rt__set_external_qreg(qreg* q) {
  qbits = xacc::as_shared_ptr(q->results());
  external_qreg_provided = true;
}

void __quantum__qis__cnot(Qubit* src, Qubit* tgt) {
  std::size_t src_copy = reinterpret_cast<std::size_t>(src);
  std::size_t tgt_copy = reinterpret_cast<std::size_t>(tgt);
  if (verbose) printf("[qir-qrt] Applying CX %lu, %lu\n", src_copy, tgt_copy);
  ::quantum::cnot({"q", src_copy}, {"q", tgt_copy});
}

void __quantum__qis__h(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying H %lu\n", qcopy);
  ::quantum::h({"q", qcopy});
}

void __quantum__qis__s(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying S %lu\n", qcopy);
  ::quantum::s({"q", qcopy});
}

void __quantum__qis__sdg(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying Sdg %lu\n", qcopy);
  ::quantum::sdg({"q", qcopy});
}
void __quantum__qis__t(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying T %lu\n", qcopy);
  ::quantum::t({"q", qcopy});
}
void __quantum__qis__tdg(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying Tdg %lu\n", qcopy);
  ::quantum::tdg({"q", qcopy});
}

void __quantum__qis__x(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying X %lu\n", qcopy);
  ::quantum::x({"q", qcopy});
}
void __quantum__qis__y(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying Y %lu\n", qcopy);
  ::quantum::y({"q", qcopy});
}
void __quantum__qis__z(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying Z %lu\n", qcopy);
  ::quantum::z({"q", qcopy});
}

void __quantum__qis__rx(double x, Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying Rx(%f) %lu\n", x, qcopy);
  ::quantum::rx({"q", qcopy}, x);
}

void __quantum__qis__ry(double x, Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying Ry(%f) %lu\n", x, qcopy);
  ::quantum::ry({"q", qcopy}, x);
}

void __quantum__qis__rz(double x, Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying Rz(%f) %lu\n", x, qcopy);
  ::quantum::rz({"q", qcopy}, x);
}
void __quantum__qis__u3(double theta, double phi, double lambda, Qubit* q) {
    std::size_t qcopy = reinterpret_cast<std::size_t>(q);
  if (verbose) printf("[qir-qrt] Applying U3(%f, %f, %f) %lu\n", theta, phi, lambda, qcopy);
  ::quantum::u3({"q", qcopy}, theta, phi, lambda);
}

Result* __quantum__qis__mz(Qubit* q) {
  if (verbose)
    printf("[qir-qrt] Measuring qubit %lu\n", reinterpret_cast<std::size_t>(q));
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);

  if (!qbits) {
    qbits = std::make_shared<xacc::AcceleratorBuffer>(allocated_qbits);
  }

  ::quantum::set_current_buffer(qbits.get());
  auto bit = ::quantum::mz({"q", qcopy});
  if (mode == QRT_MODE::FTQC)
    if (verbose) printf("[qir-qrt] Result was %d.\n", bit);
  return bit ? ResultOne : ResultZero;
}

Array* __quantum__rt__qubit_allocate_array(uint64_t size) {
  if (verbose) printf("[qir-qrt] Allocating qubit array of size %lu.\n", size);

  auto new_array = std::make_unique<Array>(size);
  for (uint64_t i = 0; i < size; i++) {
    auto qubit = new uint64_t;  // Qubit("q", i);
    *qubit = i;
    (*new_array)[i] = reinterpret_cast<int8_t*>(qubit);
  }

  allocated_qbits = size;
  if (!qbits) {
    qbits = std::make_shared<xacc::AcceleratorBuffer>(size);
    ::quantum::set_current_buffer(qbits.get());
  }

  auto raw_ptr = new_array.get();
  allocated_arrays.push_back(std::move(new_array));
  return raw_ptr;
}

int8_t* __quantum__rt__array_get_element_ptr_1d(Array* q, uint64_t idx) {
  Array& arr = *q;
  int8_t* ptr = arr[idx];
  Qubit* qq = reinterpret_cast<Qubit*>(ptr);

  if (verbose)
    printf("[qir-qrt] Returning qubit array element %lu, idx=%lu.\n", *qq, idx);
  return ptr;
}

void __quantum__rt__qubit_release_array(Array* q) {
  for (std::size_t i = 0; i < allocated_arrays.size(); i++) {
    if (allocated_arrays[i].get() == q) {
      auto& array_ptr = allocated_arrays[i];
      auto array_size = array_ptr->size();
      if (verbose && mode == QRT_MODE::FTQC)
        printf("[qir-qrt] deallocating the qubit array of size %lu\n",
               array_size);
      for (int k = 0; k < array_size; k++) {
        delete (*array_ptr)[k];
      }
      array_ptr->clear();
    }
  }
}

void __quantum__rt__finalize() {
  if (verbose) std::cout << "[qir-qrt] Running finalization routine.\n";
  if (mode == QRT_MODE::NISQ) {
    xacc::internal_compiler::execute_pass_manager();
    ::quantum::submit(qbits.get());
    auto counts = qbits->getMeasurementCounts();
    std::cout << "Observed Counts:\n";
    for (auto [bits, count] : counts) {
      std::cout << bits << " : " << count << "\n";
    }
  } else if (external_qreg_provided) {
    xacc::internal_compiler::execute_pass_manager();
    ::quantum::submit(qbits.get());
  }
}

void __quantum__qis__exp__body(Array *paulis, double angle, Array *qubits) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__exp__adj(Array *paulis, double angle, Array *qubits) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__exp__ctl(Array *ctls, Array *paulis, double angle,
                              Array *qubits) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__exp__ctladj(Array *ctls, Array *paulis, double angle,
                                 Array *qubits) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__h__body(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Delegate to __quantum__qis__h
  __quantum__qis__h(q);
}
void __quantum__qis__h__ctl(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__body(Pauli pauli, double theta, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__adj(Pauli pauli, double theta, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__ctl(Array *ctls, Pauli pauli, double theta, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__ctladj(Array *ctls, Pauli pauli, double theta,
                               Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__s__body(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__s__adj(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__s__ctl(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__s__ctladj(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__t__body(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__t__adj(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__t__ctl(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__t__ctladj(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__x__body(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__x__adj(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__x__ctl(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__x__ctladj(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__y__body(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__y__adj(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__y__ctl(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__y__ctladj(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__z__body(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__z__adj(Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__z__ctl(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__z__ctladj(Array *ctls, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__rx__body(double theta, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__ry__body(double theta, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__rz__body(double theta, Qubit *q) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__cnot__body(Qubit *src, Qubit *tgt) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__cnot(src, tgt);
}

Result *__quantum__qis__measure__body(Array *bases, Array *qubits) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}
double __quantum__qis__intasdouble__body(int32_t intVal) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return 0.0;
}
void __quantum__rt__array_update_alias_count(Array *bases, int64_t count) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}

bool __quantum__rt__result_equal(Result *res, Result *comp) {
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // std::cout << "RES = " << res << "\n";
  // std::cout << "COMP = " << comp << "\n";
  // We can do pointer comparison here.
  return res == comp;
}

int64_t __quantum__rt__array_get_size_1d(Array *state1) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return 0;
}
int8_t *__quantum__rt__tuple_create(int64_t state) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}
void __quantum__rt__string_update_reference_count(void *str, int64_t count) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__rt__array_update_reference_count(Array *aux, int64_t count) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__rt__result_update_reference_count(Result *, int64_t count) {
  // TODO
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}

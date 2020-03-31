// Build and install XACC branch mccaskey/clang_syntax_handler
// (build instructions at https://xacc.readthedocs.io/en/latest/install.html)
//
// clang++ -fplugin=$HOME/.xacc/clang-plugins/libqcor-xasm-handler.so -I $HOME/.xacc/include/xacc -c simple-bell.cpp -o bell.o
// clang++ -rdynamic -Wl,-rpath,$HOME/.xacc/lib -L $HOME/.xacc/lib -lxacc -lCppMicroServices bell.o -o bell.exe
// ./bell.exe (to see optimized circuit and resultant bit strings)

#include <qalloc>

// A quantum kernel representing a bell state
// circuit, but with too many CXs, a quantum
// compiler should be able to optimize these away
[[clang::syntax(xasm)]] void bell(qreg q) {
  // First bell state on qbits 0,1
  H(q[0]);
  // Demonstrate that our quantum
  // compiler optimizes
  CX(q[0],q[1]);
  CX(q[0],q[1]);
  CX(q[0],q[1]);

  // Measure them all
  Measure(q[0]);
  Measure(q[1]);
}

int main(int argc, char** argv) {

    // Allocate 2 qubits
    auto q = qalloc(2);

    // Execute on the quantum accelerator
    bell(q);

    // Get the results and display
    auto counts = q.counts();
    for (const auto & kv: counts) {
        printf("%s: %i\n", kv.first.c_str(), kv.second);
    }
}

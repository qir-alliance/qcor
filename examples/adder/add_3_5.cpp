
__qpu__ void add_3_5(qreg a, qreg b, qreg c) {
  using qcor::openqasm;

  oracle adder a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3 { "add_3_5.v" }

  creg result[4];
  // a = 3
  x a[0];
  x a[1];

  // b = 5
  x b[0];
  x b[2];

  adder a[0],a[1],a[2],a[3],b[0],b[1],b[2],b[3],c[0],c[1],c[2],c[3];

  // measure
  measure c -> result;
}

int main(int argc, char** argv) {

    auto a = qalloc(4);
    auto b = qalloc(4);
    auto c = qalloc(4);

    // Execute on the quantum accelerator
    add_3_5(a, b, c);

    // Get the results and display
    auto counts = c.counts();
    for (const auto & kv: counts) {
        printf("%s: %i\n", kv.first.c_str(), kv.second);
    }
}

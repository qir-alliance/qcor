// X kernel
__qpu__ void x_gate(qreg q) { X(q[0]); }

__qpu__ void ccxtest(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    X(q[i]);
  } 

  // apply ctrl-U (CCX = Toffoli)
  x_gate::ctrl({q[1], q[2]}, q);

  // measure
  for (int i = 0; i < q.size(); i++) {
    Measure(q[i]);
  } 
}

int main() {
  auto q = qalloc(3);
  ccxtest(q);
  q.print();
}
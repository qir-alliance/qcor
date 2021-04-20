__qpu__ void bell(qubit q, qubit r) {
  H(q);
  X::ctrl(q,r);
  Measure(q);
  Measure(r);
}

int main() {
  auto q = qalloc(1);
  auto r = qalloc(1);
  bell(q[0],r[0]);
  q.print();
  r.print();
}
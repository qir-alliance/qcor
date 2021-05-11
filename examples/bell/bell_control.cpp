__qpu__ void bell(qubit q, qubit r) {
  H(q);
  X::ctrl(q,r);
  Measure(q);
  Measure(r);
}

int main() {
  set_shots(1024);
  auto q = qalloc(1);
  auto r = qalloc(1);
  bell(q[0],r[0]);
  q.print();
  r.print();
  qcor_expect(q.counts().size() == 2);
  qcor_expect(r.counts().size() == 2);
  qcor_expect(q.counts()["0"] == r.counts()["0"]);
  qcor_expect(q.counts()["1"] == r.counts()["1"]);
}
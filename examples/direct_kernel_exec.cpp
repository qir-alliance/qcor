#include <qalloc>

__qpu__ void ansatz(qreg q, double t) {
  X(q[0]);
  Ry(q[1], t);
  CX(q[1], q[0]);
  H(q);
  Measure(q);
}

int main(int argc, char **argv) {
  auto q = qalloc(2);
  double * angles = new double[5] {-3.14,1.57,0.0,1.57,3.14};
  for (int i = 0; i < 5; i++) {
      ansatz(q, angles[i]);
      printf("<X0X1>(%f) = %f \n", angles[i],  q.exp_val_z());
  }
}

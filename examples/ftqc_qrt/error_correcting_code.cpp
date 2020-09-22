#include <qcor_qec>

int main() {
  auto q = qalloc(4);
  bit_flip_encoder(q, 0, {1, 2});
}
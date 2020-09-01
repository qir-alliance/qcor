// run error mitigation with mitiq with 
// $ qcor -qpu aer[noise-model:noise_model.json] -shots 4096 -em mitiq simple_mitiq.cpp
// $ ./a.out

__qpu__ void noisy_zero(qreg q) {
    for (int i = 0; i < 100; i++) {
        X(q[0]);
    }
    Measure(q[0]);
}

int main() {
    qreg q = qalloc(1);
    noisy_zero(q);
    std::cout << "Expectation: " << q.exp_val_z() << "\n";
}
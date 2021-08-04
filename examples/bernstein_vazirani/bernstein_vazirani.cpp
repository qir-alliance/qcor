__qpu__ void bernstein_vazirani(qreg q, std::string &secret_bits) {
    // prepare ancilla in |1>
    X(q[secret_bits.size()]);

    // input superpositions
    for (int i = 0; i <= secret_bits.size(); i++) {
        H(q[i]);
    }

    // oracle
    for (int i = 0; i < secret_bits.size(); i++) {
        if (secret_bits[i] == '1') {
            CX(q[i], q[secret_bits.size()]);
        }
    }

    for (int i = 0; i <= secret_bits.size(); i++) {
        H(q[i]);

        if (i < secret_bits.size())
            Measure(q[i]);
    }
}

int main() {
    set_shots(1024);
    std::string secret_bits = "110101";

    auto q = qalloc(secret_bits.size() + 1);
    bernstein_vazirani(q, secret_bits);
    q.print();

    qcor_expect(q.counts().size() == 1);
    qcor_expect(q.counts()[secret_bits] == 1024);
}

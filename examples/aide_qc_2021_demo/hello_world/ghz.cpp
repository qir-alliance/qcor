// Demonstrate the programmability of a simple kernel
// Run with QPP first
// Run with Aer noisy backend
// Run with just print_kernel to show automatic placement
// Run on IBM

__qpu__ void ghz(qreg q) {
    auto first = q.head();

    H(first);

    for (int i = 0; i < q.size()-1; i++) {
        CX(q[i], q[i+1]);
    }

    Measure(q);
}

int main() {

    set_shots(100);
    auto q = qalloc(6);

    ghz(q);

    auto counts = q.counts();

    for (auto [bit, count] : counts) {
        print(bit, ": ", count);
    }

    ghz::print_kernel(q);
}
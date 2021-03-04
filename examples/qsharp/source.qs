namespace XACC 
{
open Microsoft.Quantum.Intrinsic;
operation TestBell(count : Int) : Int {
    // Simple bell test
    mutable numOnes = 0;
    use q = Qubit[2];
    for test in 1..count {
        H(q[0]);
        CNOT(q[0],q[1]);
        let res = M(q[0]);

        // Count the number of ones we saw:
        if res == One {
            set numOnes += 1;
            X(q[0]);
            X(q[1]);
        }
    }
    return numOnes;
}
}
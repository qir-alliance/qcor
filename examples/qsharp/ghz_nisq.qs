namespace XACC 
{
open Microsoft.Quantum.Intrinsic;
operation TestGhz(q : Qubit[]) : Unit {
    H(q[0]);
    CNOT(q[0],q[1]);
    CNOT(q[0],q[2]);
    let res0 = M(q[0]);
    let res1 = M(q[1]);
    let res2 = M(q[2]);
}
}
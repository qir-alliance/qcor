namespace QCOR 
{
open QCOR.Intrinsic;
operation TestKernel(q : Qubit[], theta: Double) : Unit {
    H(q[0]);
    Ry(theta, q[1]);
    CNOT(q[0],q[1]);
    CNOT(q[0],q[2]);
    let res0 = M(q[0]);
    let res1 = M(q[1]);
    let res2 = M(q[2]);
}
}
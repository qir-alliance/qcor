namespace QCOR 
{
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Canon;

operation SWAP(q1 : Qubit, q2: Qubit) : Unit is Adj {
  CNOT(q1, q2);
  CNOT(q2, q1);
  CNOT(q1, q2);
}

operation IQFT(qq: Qubit[]): Unit {
  Message("Hello from Q# IQFT");
  for i in 0 .. Length(qq)/2 - 1 {
    SWAP(qq[i], qq[Length(qq)-i-1]);
  }
    
  for i in 0 .. Length(qq) - 2 {
    H(qq[i]);
    let j = i + 1;
    mutable y = i;
    repeat {
      let theta = -3.14159 / IntAsDouble(1 <<< (j-y));
      // Controlled R1 == CPhase
      Controlled R1([qq[j]], (theta, qq[y]));
      set y = y - 1;
    } until (y < 0);
  }

  H(qq[Length(qq) -1]);
}
}
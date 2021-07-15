namespace QCOR 
{
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Canon;

operation ApplyKernelToEachQubit(singleElementOperation : (Qubit => Unit is Adj + Ctl)): Unit {
  use q = Qubit[5];
  ApplyToEachCA(singleElementOperation, q);
  for idx in 0..4 {
    let res = M(q[idx]);    
    if res == One {
      Message("Get one");
    } else {
      Message("Get zero");
    }
  }
}
}
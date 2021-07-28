namespace QCOR 
{
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Canon;

// operation ApplyKernelToEachQubit(singleElementOperation : (Qubit => Unit is Adj + Ctl)): Unit {
//   use q = Qubit[5];
//   ApplyToEachCA(singleElementOperation, q);
//   for idx in 0..4 {
//     let res = M(q[idx]);    
//     if res == One {
//       Message("Get one");
//     } else {
//       Message("Get zero");
//     }
//   }
// }

operation ApplyControlledKernel(singleElementOperation : ((Qubit) => Unit is Adj + Ctl)): Unit {
  use q = Qubit[2];
  
  H(q[0]);
  // Apply controlled
  Controlled singleElementOperation([q[0]], (q[1])); 
  let res0 = M(q[0]);    
  if res0 == One {
    Message("Meas Q0 -> one");
    X(q[0]);
  } else {
    Message("Meas Q0 -> zero");
  }

  let res1 = M(q[1]);    
  if res1 == One {
    Message("Meas Q1 -> one");
    X(q[1]);
  } else {
    Message("Meas Q0 -> zero");
  }
}
}
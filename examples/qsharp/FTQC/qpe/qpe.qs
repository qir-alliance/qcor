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

operation Rotation (q : Qubit, angle: Double) : Unit is Adj+Ctl {
  R1(angle, q);
}

operation IQFT(qq: Qubit[]): Unit {
  for i in 0 .. Length(qq)/2 - 1 {
    SWAP(qq[i], qq[Length(qq)-i-1]);
  }
    
  for i in 0 .. Length(qq) - 2 {
    H(qq[i]);
    let j = i + 1;
    mutable y = i;
    repeat {
      let theta = -3.14159 / IntAsDouble(1 <<< (j-y));
      Controlled Rotation([qq[j]], (qq[y], theta));
      set y = y - 1;
    } until (y < 0);
  }

  H(qq[Length(qq) -1]);
}

operation t_oracle (qb: Qubit) : Unit is Adj+Ctl {
  // Oracle = T gate
  T(qb);
}

@EntryPoint() 
operation QuantumPhaseEstimation(): Int {
  // 3-qubit QPE:
  use (counting, state) = (Qubit[3], Qubit()) 
  {
    // We want T |1> = exp(2*i*pi*phase) |1> = exp(i*pi/4)
    // compute phase, should be 1 / 8;
    // Initialize to |1>
    X(state);
    // Put all others in a uniform superposition
    // Use this Q# equiv. of broadcast:
    ApplyToEach(H, counting);
    mutable repetitions = 1;
    for i in 0 .. Length(counting) - 1 {
      // Loop over and create ctrl-U**2k
      for j in 1 .. repetitions {
        Controlled t_oracle([counting[i]], state);
      }
      set repetitions = repetitions * 2;
    }

    // Run IQFT 
    IQFT(counting);
    mutable result = 0; 
    // Now lets measure the counting qubits
    // Convert it to MSB ==> expect 4
    for i in 0 .. Length(counting) - 1 {
      if (M(counting[i]) == One) {
        set result = result + (1 <<< (Length(counting) - i - 1));
        X(counting[i]);
      }
    }
    Message($"Result = {result}");
    return result;
  }
}
}
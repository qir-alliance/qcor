namespace QCOR 
{
open Microsoft.Quantum.Intrinsic;

operation GenerateRandomInt(maxBits: Int): Int {
  mutable rngNumber = 0;
  use qubit = Qubit() 
  {
    for idx in 1..maxBits {
      H(qubit);
      if (M(qubit) == One) {
        set rngNumber = rngNumber + (1 <<< (idx - 1));
        // Reset
        X(qubit);
      }
    }
  }
  Message($"Random number from Q# = {rngNumber}");
  return rngNumber;
}
}


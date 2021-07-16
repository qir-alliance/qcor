namespace QCOR {
  open Microsoft.Quantum.Math;
  open Microsoft.Quantum.Canon;
  open Microsoft.Quantum.Intrinsic;
  open Microsoft.Quantum.Characterization;
  
  /// Given two operations which each prepare copies of a state, estimates
  /// the real part of the overlap between the states prepared by each
  /// operation.
  /// Leverage Q# Characterization library: 
  operation ComputeOverlapBetweenState(preparation1 : (Qubit => Unit is Adj + Ctl), preparation2 : (Qubit => Unit is Adj + Ctl), nMeasurements : Int) : Double {
    // Can use Q# to convert the functor types as well
    let prep1 = ApplyToEachCA(preparation1, _);
    let prep2 = ApplyToEachCA(preparation2, _);
    // This function is part of the Q# standard library
    return EstimateRealOverlapBetweenStates(NoOp<Qubit[]>, prep1, prep2, 1, nMeasurements);
  }
}
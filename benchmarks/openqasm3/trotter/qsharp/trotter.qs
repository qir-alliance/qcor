namespace Benchmark.Heisenberg {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Measurement;
    // In this example, we will show how to simulate the time evolution of
    // an Heisenberg model: 
    //     H ≔ - J Σ'ᵢⱼ Zᵢ Zⱼ - hX Σᵢ Xᵢ
    // where the primed summation Σ' is taken only over nearest-neighbors.
    operation HeisenbergTrotterEvolve(nSites : Int, simulationTime : Double, trotterStepSize : Double) : Unit {
        // We pick arbitrary values for the X and J couplings
        let hXCoupling = 1.0;
        let jCoupling = 1.0;

        // This determines the number of Trotter steps
        let steps = Ceiling(simulationTime / trotterStepSize);

        // This resizes the Trotter step so that time evolution over the
        // duration is accomplished.
        let trotterStepSizeResized = simulationTime / IntAsDouble(steps);

        // Let us initialize nSites clean qubits. These are all in the |0>
        // state.
        use qubits = Qubit[nSites];
        // We then evolve for some time
        for idxStep in 0 .. steps - 1 {
            for i in 0 .. nSites - 1 {
                Exp([PauliX], (-1.0 * hXCoupling) * trotterStepSizeResized, [qubits[i]]);
            }
            for i in 0 .. nSites - 2 {
                Exp([PauliZ, PauliZ], (-1.0 * jCoupling) * trotterStepSizeResized, qubits[i .. (i + 1)]);
            }
            
        }
    }

    // Entry point: we allow the Q# program to have full information (compile-time) about the number of qubits, steps, etc.
    // to be equivalent to OpenQASM3
    @EntryPoint()
    operation CircuitGen() : Unit {
        HeisenbergTrotterEvolve(50, 1.0, 0.01);
    }
}
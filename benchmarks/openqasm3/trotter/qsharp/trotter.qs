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

    operation _Heisenberg1DTrotterUnitaries(nSites : Int, hXCoupling : Double, jCoupling : Double, idxHamiltonian : Int, stepSize : Double, qubits : Qubit[])
    : Unit
    is Adj + Ctl {
        // when idxHamiltonian is in [0, nSites - 1], apply transverse field "hx"
        // when idxHamiltonian is in [nSites, 2 * nSites - 1], apply Heisenberg coupling "jC"
        if (idxHamiltonian <= nSites - 1) {
            Exp([PauliX], (-1.0 * hXCoupling) * stepSize, [qubits[idxHamiltonian]]);
        } 
        else {
            Exp([PauliZ, PauliZ], (-1.0 * jCoupling) * stepSize, qubits[idxHamiltonian % nSites .. (idxHamiltonian + 1) % nSites]);
        }
    }

    // The input to the Trotterization control structure has a type
    // (Int, ((Int, Double, Qubit[]) => () is Adj + Ctl))
    // The first parameter Int is the number of terms in the Hamiltonian
    // The first parameter in ((Int, Double, Qubit[])) is an index to a term
    // in the Hamiltonian
    // The second parameter in ((Int, Double, Qubit[])) is the stepsize
    // The third parameter in  ((Int, Double, Qubit[])) are the qubits the
    // Hamiltonian acts on.
    // Let us create this type from Heisenberg1DTrotterUnitariesImpl by partial
    // applications.
    function Heisenberg1DTrotterUnitaries(nSites : Int, hXCoupling : Double, jCoupling : Double)
    : (Int, ((Int, Double, Qubit[]) => Unit is Adj + Ctl)) {
        let nTerms = 2 * nSites - 1;
        return (nTerms, _Heisenberg1DTrotterUnitaries(nSites, hXCoupling, jCoupling, _, _, _));
    }

    // We now invoke the Trotterization control structure. This requires two
    // additional parameters -- the trotterOrder, which determines the order
    // the Trotter decompositions, and the trotterStepSize, which determines
    // the duration of time-evolution of a single Trotter step.
    function Heisenberg1DTrotterEvolution(nSites : Int, hXCoupling : Double, jCoupling : Double, trotterOrder : Int, trotterStepSize : Double)
    : (Qubit[] => Unit is Adj + Ctl) {
        let op = Heisenberg1DTrotterUnitaries(nSites, hXCoupling, jCoupling);
        return DecomposedIntoTimeStepsCA(op, trotterOrder)(trotterStepSize, _);
    }

    operation HeisenbergTrotterEvolve(nSites : Int, simulationTime : Double, trotterOrder : Int, trotterStepSize : Double) : Unit {
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
            Heisenberg1DTrotterEvolution(nSites, hXCoupling, jCoupling, trotterOrder, trotterStepSizeResized)(qubits);
        }
    }

    // Entry point: we allow the Q# program to have full information (compile-time) about the number of qubits, steps, etc.
    // to be equivalent to OpenQASM3
    @EntryPoint()
    operation CircuitGen() : Unit {
        HeisenbergTrotterEvolve(5, 1.0, 1, 0.01);
    }
}
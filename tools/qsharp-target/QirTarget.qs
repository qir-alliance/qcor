// Define QCOR Target instruction set
namespace QCOR.Instructions {
    operation X(qb : Qubit) : Unit {
        body intrinsic;
    }
    operation Y(qb : Qubit) : Unit {
        body intrinsic;
    }

    operation Z(qb : Qubit) : Unit {
        body intrinsic;
    }

    operation H(qb : Qubit) : Unit {
        body intrinsic;
    }

    operation Rx (theta : Double, qb : Qubit) : Unit {
        body intrinsic;
    }

    operation Ry (theta : Double, qb : Qubit) : Unit {
        body intrinsic;
    }
    operation Rz (theta : Double, qb : Qubit) : Unit {
        body intrinsic;
    }

    operation Reset(qb : Qubit) : Unit {
        body intrinsic;
    }

    operation CNOT(ctrl : Qubit, target : Qubit) : Unit {
        body intrinsic;
    }
}

// Define QCOR Intrinsic
namespace QCOR.Intrinsic {
    open Microsoft.Quantum.Targeting;
    open QCOR.Instructions as Phys;
    
    @Inline()
    function PI() : Double
    {
        return 3.14159265357989;
    }
    
    function IntAsDouble(i : Int) : Double {
        body intrinsic;
    }

    operation X(qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.X(qb);  
        }
        adjoint self;
    }

    operation Y(qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.Y(qb);  
        }
        adjoint self;
    }

    operation Z(qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.Y(qb);  
        }
        adjoint self;
    }

   operation H(qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.H(qb);  
        }
        adjoint self;
    }

    @Inline()
    operation S(qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.Rz(PI()/2.0, qb);  
        }
        adjoint (...)
        {
            Phys.Rz(-PI()/2.0, qb); 
        }
    }


    @Inline()
    operation T(qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.Rz(PI()/4.0, qb);  
        }
        adjoint (...)
        {
            Phys.Rz(-PI()/4.0, qb); 
        }
    }
   

    operation CNOT(control : Qubit, target : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.CNOT(control, target);  
        }
        adjoint self;
    }

    @TargetInstruction("mz")
    operation M(qb : Qubit) : Result {
        body intrinsic;
    }

    operation Measure(bases : Pauli[], qubits : Qubit[]) : Result {
        body intrinsic;
    }

    operation MResetZ(qb : Qubit) : Result
    {
        let res = M(qb);
        Phys.Reset(qb); 
        return res;
    }

    @Inline()
    operation Rx(theta : Double, qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.Rx(theta, qb);  
        }
        adjoint (...)
        {
            Phys.Rx(-theta, qb);  
        }
    }

    @Inline()
    operation Ry(theta : Double, qb : Qubit) : Unit
    is Adj {
        body  (...)
        {
            Phys.Ry(theta, qb);  
        }
        adjoint (...)
        {
            Phys.Ry(-theta, qb);  
        }
    }

    @Inline()
    operation Rz(theta : Double, qb : Qubit) : Unit
    is Adj + Ctl {
        body  (...)
        {
            Phys.Rz(theta, qb);  
        }
        adjoint (...)
        {
            Phys.Rz(-theta, qb);  
        }
        controlled (ctls, ...)
        {
            Phys.Rz(theta / 2.0, qb);
            CNOT(ctls[0], qb);
            Phys.Rz(-theta / 2.0, qb);
            CNOT(ctls[0], qb);
        }
        controlled adjoint (ctls, ...)
        {
            Phys.Rz(-theta / 2.0, qb);
            CNOT(ctls[0], qb);
            Phys.Rz(theta / 2.0, qb);
            CNOT(ctls[0], qb);
        }
    }

    @Inline()
    operation Reset(qb : Qubit) : Unit {
        body  (...)
        {
            Phys.Reset(qb);  
        }
    }
}
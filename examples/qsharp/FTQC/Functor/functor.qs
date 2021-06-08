namespace QCOR.Testing {
    open Microsoft.Quantum.Intrinsic;
    operation Qop(q : Qubit, n : Int) : Unit
    is Adj+Ctl {
        body (...) {
            if n%2 == 1 { X(q); }
        }
        adjoint self;
        controlled (ctrls, ...) {
            if n%2 == 1 { Controlled X(ctrls, q); }
        }
    }

    @EntryPoint()
    operation TestFunctors() : Unit {
        let qop = Qop(_, 1);
        let adj_qop = Adjoint qop;
        let ctl_qop = Controlled qop;
        let adj_ctl_qop = Adjoint Controlled qop;
        let ctl_ctl_qop = Controlled ctl_qop;

        use (q1, q2, q3) = (Qubit(), Qubit(), Qubit()) {
            qop(q1);
            if (M(q1) != One) { Message("error code: 1"); }
 
            adj_qop(q2);
            if (M(q2) != One) { Message("error code: 2"); }

            ctl_qop([q1], q3);
            if (M(q3) != One) { Message("error code: 3"); }

            adj_ctl_qop([q2], q3);
            if (M(q3) != Zero) { Message("error code: 2"); }

            ctl_ctl_qop([q1], ([q2], q3));
            if (M(q3) != One) { Message("error code: 5"); }

            Controlled qop([q1, q2], q3);
            if (M(q3) != Zero) { Message("error code: 6"); }

            use q4 = Qubit() {
                Adjoint qop(q3);
                Adjoint Controlled ctl_ctl_qop([q1], ([q2], ([q3], q4)));
                if (M(q4) != One) { Message("error code: 7"); }
            }
        }
    }

    operation NoArgs() : Unit
    is Adj+Ctl {
        body (...) {
            use q = Qubit();
            X(q);
        }
        adjoint self;
        controlled (ctrls, ...) {
            use q = Qubit();
            Controlled X(ctrls, q);
        }
    }

    @EntryPoint()
    operation TestFunctorsNoArgs() : Unit {
        NoArgs();
        let qop = NoArgs;
        let adj_qop = Adjoint qop;
        let ctl_qop = Controlled qop;
        let adj_ctl_qop = Adjoint Controlled qop;
        let ctl_ctl_qop = Controlled ctl_qop;

        use (q1, q2, q3) = (Qubit(), Qubit(), Qubit()) {
            X(q1);
            X(q2);
            X(q3);

            qop();
            adj_qop();
            ctl_qop([q1], ());
            adj_ctl_qop([q1], ());
            ctl_ctl_qop([q1], ([q2], ()));

            Controlled qop([q1, q2], ());
            Adjoint Controlled ctl_ctl_qop([q1], ([q2], ([q3], ())));
        }
    }
}
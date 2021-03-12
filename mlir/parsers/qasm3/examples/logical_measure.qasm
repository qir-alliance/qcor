OPENQASM 3;
include "stdgates.inc";

// declarations
const n = 3;

//kernel vote(bit[n]) -> bit;
def vote(bit[n]:c) -> bit {
    int[32] vote_count = 0;
    for i in [0:n] {
        if (c[i] == 1) {
            vote_count += 1;
        }
    }

    bit ret;
    if (vote_count > 1) {
        ret = 1;
    } else {
        ret = 0;
    }
    return ret;
}

def logical_meas qubit[3]:d -> bit {
    bit c[3];
    bit r;
    measure d -> c;
    //r = vote(c);
    return r;
}

qubit q[3];
qubit a[3];
bit r;

// prep magic state
rz(pi/4) a;

// entangle two logical registers
cx q, a;

// measure out the ancilla
r = logical_meas a;

// if we get a logical |1> then we need to apply a logical Z correction
if (r == 1) z q;
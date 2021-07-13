OPENQASM 3;

def cnot_ladder(bit[4]:operator) qubit[4]:q {

  for i in [0:2] {
      if (operator[i] == operator[i+2] && operator[i] != 1) {
          rx(1.57) q[i];
      } else if (operator[i] == 1 && operator[i+2] == 0) {
          h q[i];
      } 
  }

  for i in [0:3] {
      cx q[i], q[i+1];
  }

}

def trotter(int[64]:n_steps, double:theta, bit[4]:operator) qubit[4]:qq {

  for i in [0:n_steps] {
      cnot_ladder(operator) qq;
      rz(theta) qq[3];
      inv @ cnot_ladder(operator) qq;
  }

}


bit x0x1[4] = "1100";
qubit r[4];

double dt = .01;
int[64] n_steps = 100;

trotter(n_steps, dt, x0x1) r;
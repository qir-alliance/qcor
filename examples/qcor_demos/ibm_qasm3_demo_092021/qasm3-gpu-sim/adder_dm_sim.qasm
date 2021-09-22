/*
 * quantum ripple-carry adder
 * Cuccaro et al, quant-ph/0410184
 */
// Install DM-SIM plugin: 
// qcor -install-plugin https://github.com/ORNL-QCI/DM-Sim.git
// Using DM-Sim:
// qcor -linker g++ -qrt nisq adder.qasm -shots 1024 -qpu dm-sim[gpus:1]
// Note: on a login node, GPU-GPU comm is disabled, hence can only run with 1 GPU.

// Multi-GPU on compute node: e.g, see the bsub example:
OPENQASM 3;

gate ccx a,b,c
{
  h c;
  cx b,c; tdg c;
  cx a,c; t c;
  cx b,c; tdg c;
  cx a,c; t b; t c; h c;
  cx a,b; t a; tdg b;
  cx a,b;
}

gate majority a, b, c {
  cx c, b;
  cx c, a;
  ccx a, b, c;
}

gate unmaj a, b, c {
  ccx a, b, c;
  cx c, a;
  cx a, b;
}

qubit cin;
qubit a[4];
qubit b[4];
qubit cout;
bit ans[5];
// Input values:
uint[4] a_in = 1;  
uint[4] b_in = 15; 

for i in [0:4] {  
  if (bool(a_in[i])) {
    x a[i];
  }
  if (bool(b_in[i])) {
    x b[i];
  }
}
// add a to b, storing result in b
majority cin, b[0], a[0];

for i in [0: 3] { 
  majority a[i], b[i + 1], a[i + 1]; 
}

cx a[3], cout;

for i in [2: -1: -1] { 
  unmaj a[i], b[i+1], a[i+1]; 
}
unmaj cin, b[0], a[0];

measure b[0:3] -> ans[0:3];
measure cout[0] -> ans[4];
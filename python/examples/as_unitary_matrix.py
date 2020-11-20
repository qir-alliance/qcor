from qcor import *
import numpy as np

@qjit
def ansatz(q : qreg, x : float):
    X(q[0])
    Ry(q[1], x)
    CX(q[1], q[0])
        
# Can convert qjit kernels to a unitary matrix form
# I know the ground state is at x = .59
u_mat = ansatz.as_unitary_matrix(qalloc(2), .59)

# Create a Hamiltonian, map it to a numpy matrix
H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
Hmat = H.to_numpy()

# Compute |psi> = U |0>
zero_state = np.array([1., 0., 0., 0.])
final_state = np.dot(u_mat, np.transpose(zero_state))

# Compute E = <psi| H |psi>
energy = np.dot(final_state, np.dot(Hmat,final_state))
print('Energy: ', energy)
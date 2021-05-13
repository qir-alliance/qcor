#pragma once

#include "qcor.hpp"
#include <qcor_qft>
// Note: We cannot put this in a namespace yet
// because we *do* want the XASM to recognize this as
// a quantum kernel hence invoking the token collector
// where we handle nested kernel calls (
// e.g. reset the __execute flag so that only the top-level
// kernel is submitted.

// Classical helper functions: wrapped it in a namespace to bypass XASM.
// These functions are used to construct circuit parameters.
namespace qcor { namespace util {
template <typename T>
T modpow(T base, T exp, T modulus) {
  base %= modulus;
  T result = 1;
  while (exp > 0) {
    if (exp & 1) result = (result * base) % modulus;
    base = (base * base) % modulus;
    exp >>= 1;
  }
  return result;
}

// Generates a list of angles to perform addition by a in the Fourier space.
void genAngles(std::vector<double>& io_angles, int a, int nbQubits) {
  // Makes sure the vector appropriately sized
  io_angles.resize(nbQubits);
  std::fill(io_angles.begin(), io_angles.end(), 0.0);

  for (int i = 0; i < nbQubits; ++i) {
    int bitIdx = nbQubits - i - 1;
    auto& angle = io_angles[bitIdx];
    for (int j = i; j < nbQubits; ++j) {
      // Check bit
      int bitShift = nbQubits - j - 1;
      if (((1 << bitShift) & a) != 0) {
        angle += std::pow(2.0, -(j-i));
      }
    }
    angle *= M_PI;
  }
}

// Calculate 2^i * a % N;
inline void calcPowMultMod(int& result, int i, int a, int N) {
  result = (1 << i) * a % N;
}

// Compute extended greatest common divisor of two integers
inline std::tuple<int, int, int> egcd(int a, int b) {
  int m, n, c, q, r;
  int m1 = 0, m2 = 1, n1 = 1, n2 = 0;

  while (b) {
    q = a / b, r = a - q * b;
    m = m2 - q * m1, n = n2 - q * n1;
    a = b, b = r;
    m2 = m1, m1 = m, n2 = n1, n1 = n;
  }

  c = a, m = m2, n = n2;

  // correct the signs
  if (c < 0) {
    m = -m;
    n = -n;
    c = -c;
  }

  return std::make_tuple(m, n, c);
}

// Modular inverse of a mod p
inline void modinv(int& result, int a, int p) {
  int x, y;
  int gcd_ap;
  std::tie(x, y, gcd_ap) = egcd(p, a);
  result = (y > 0) ? y : y + p;
}

// Compute the minimum number of bits required
// to represent an integer number N.
inline void calcNumBits(int& result, int N) {
  int count = 0;
  while (N) {
    count++;
    N = N >> 1;
  }
  result = count;
}

// Compute a^(2^i) mod N
inline void calcExpExp(int& result, int a, int i, int N) {
  result = modpow(a, 1 << i, N);
}
}}

// Addition by a in Fourier Space
// If inverse != 0, run the reverse (minus)
// Note: the number to be added needs to be converted to an array of angles
// before calling this.
// Note: we supply the start Idx so that we can specify a subset of the register.
// Param: a -> the number to add
// startBitIdx and nbQubits specify a contiguous section of the qreg.
__qpu__ void phiAdd(qreg q, int a, int startBitIdx, int nbQubits, int inverse) {
  std::vector<double> angles;
  qcor::util::genAngles(angles, a, nbQubits);   
  for (int i = 0; i < nbQubits; ++i) {
    double theta = (inverse == 0) ? angles[i] : -angles[i];
    int idx = startBitIdx + i;
    U1(q[idx], theta);
  }
}

// Addition by a number (a) in Fourier Space
__qpu__ void phi_add(qreg q, int a) {
  std::vector<double> angles;
  qcor::util::genAngles(angles, a, q.size());
  for (int i = 0; i < q.size(); ++i) {
    U1(q[idx], angles[i]);
  }
}

__qpu__ void controlled_phi_add(qreg q, int a, std::vector<qubit> ctrl_qubits) {
  return phi_add::ctrl(ctrl_qubits, q, a);
}

__qpu__ void cc_phi_add_modN(qreg q, qubit ctl1, int qubit, qubit aux, int a,
                             int N) {
  //ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv0);
  controlled_phi_add(q, a, {ctl1, ctl2});
  
  // phiAdd(q, N, startBitIdx, nbQubits, inv1); 
  // Inverse
  phi_add::adjoint(q, N);
  
  
  // iqft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  iqft(q);
  
  //int lastIdx = startBitIdx + nbQubits - 1;
  // CX(q[lastIdx], q[aux]);
  CX(q.tail(), aux);
  
  //qft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  qft(q);
  
  // cPhiAdd(q, aux, N, startBitIdx, nbQubits, inv0);
  controlled_phi_add(q, N, {aux});

  //ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv1);
  // Inverse
  controlled_phi_add::adjoint(q, a, {ctl1, ctl2});

  // iqft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  iqft(q);
  
  //X(q[lastIdx]);
  X(q.tail());
  //CX(q[lastIdx], q[aux]);
  CX(q.tail(), aux);
  // X(q[lastIdx]);
  X(q.tail());

  // qft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  qft(q);

  // ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv0);
  controlled_phi_add::adjoint(q, a, {ctl1, ctl2});
}

__qpu__ void cMultModN(qreg q, qubit ctrlIdx, int a, int N, qreg aux_reg) {
  // QFT on the auxilary:
  qft(aux_reg);
  for (int i = 0; i < q.size(); ++i) {
    int tempA = 0;
    qcor::util::calcPowMultMod(tempA, i, a, N);
    qubit auxBit = aux_reg.tail();
    // Doubly-controlled modular add on the Aux register.
    cc_phi_add_modN(aux_reg.head(aux_reg.size() - 1), q[i], ctrlIdx, auxBit, tempA, N);
  }

  iqft(aux_reg);
  
  for (int i = 0; i < q.size(); ++i) {
    Swap::ctrl(ctrlIdx, q[i], aux_reg[i]);
  }

  qft(aux_reg);
  
  int aInv = 0;
  qcor::util::modinv(aInv, a, N);
  for (int i = q.size() - 1; i >= 0; --i) {
    int tempA = 0;
    qcor::util::calcPowMultMod(tempA, i, aInv, N);
    int xIdx = startBitIdx + i;
    int auxBit = auxStartBitIdx + auxNbQubits - 1;
    cc_phi_add_modN::adjoint(aux_reg.head(aux_reg.size() - 1), q[i], ctrlIdx, auxBit, tempA, N);
  }

  iqft(aux_reg);
}

// Controlled add in Fourier Space
__qpu__ void cPhiAdd(qreg q, int ctrlBit, int a, int startBitIdx, int nbQubits, int inverse) {
  std::vector<double> angles;
  qcor::util::genAngles(angles, a, nbQubits);  
  
  for (int i = 0; i < nbQubits; ++i) {
    double theta = (inverse == 0) ? angles[i] : -angles[i] ;
    int idx = startBitIdx + i;
    // Note: ctrlBit must be different from idx,
    // i.e. ctrlBit must not be in the bitIdx vector
    // We use CPhase which is equivalent with IBM cu1
    CPhase(q[ctrlBit], q[idx], theta);
  }
}

__qpu__ void ccPhase(qreg q, int ctl1, int ctl2, int tgt, double angle) {
  CPhase(q[ctl1], q[tgt], angle/2);
  CX(q[ctl2], q[ctl1]);
  CPhase(q[ctl1], q[tgt], -angle/2);
  CX(q[ctl2], q[ctl1]);
  CPhase(q[ctl2], q[tgt], angle/2);
}

// Doubly-controlled Add operation in Fourier space
__qpu__ void ccPhiAdd(qreg q, int ctrlBit1, int ctrlBit2, int a, int startBitIdx, int nbQubits, int inverse) {
  std::vector<double> angles;
  qcor::util::genAngles(angles, a, nbQubits);  
  
  for (int i = 0; i < nbQubits; ++i) {
    double theta = (inverse == 0) ? angles[i] : -angles[i] ;
    int idx = startBitIdx + i;
    ccPhase(q, ctrlBit1, ctrlBit2, idx, theta);
  }
}

// Doubly-controlled *modular* addition by 
// Inputs: 
// - N the mod factor (represented as angles)
// - a the a addition operand (represented as angles)
__qpu__ void ccPhiAddModN(qreg q, int ctl1, int ctl2, int aux,
                          int a, int N,
                          int startBitIdx, int nbQubits) {
  // FIXME: XASM should allow literal values in function calls
  int inv0 = 0;
  int inv1 = 1;
  int withSwap = 0;
  ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv0);
  phiAdd(q, N, startBitIdx, nbQubits, inv1);
  iqft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  int lastIdx = startBitIdx + nbQubits - 1;
  CX(q[lastIdx], q[aux]);
  qft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  cPhiAdd(q, aux, N, startBitIdx, nbQubits, inv0);
  ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv1);
  iqft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  X(q[lastIdx]);
  CX(q[lastIdx], q[aux]);
  X(q[lastIdx]);
  qft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv0);                          
}

// Inverse doubly-controlled Modular Addition (Fourier space)
__qpu__ void ccPhiAddModN_inv(qreg q, int ctl1, int ctl2, int aux,
                          int a, int N,
                          int startBitIdx, int nbQubits) {
  // FIXME: XASM should allow literal values in function calls
  int inv0 = 0;
  int inv1 = 1;
  int withSwap = 0;
  ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv1);                          
  iqft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  int lastIdx = startBitIdx + nbQubits - 1;
  X(q[lastIdx]);
  CX(q[lastIdx], q[aux]);
  X(q[lastIdx]);
  qft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv0);
  cPhiAdd(q, aux, N, startBitIdx, nbQubits, inv1);
  iqft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  CX(q[lastIdx], q[aux]);
  qft_range_opt_swap(q, startBitIdx, nbQubits, withSwap);
  phiAdd(q, N, startBitIdx, nbQubits, inv0);
  ccPhiAdd(q, ctl1, ctl2, a, startBitIdx, nbQubits, inv1);
}

// Swap kernel: to be used to generate Controlled-Swap kernel
__qpu__ void SwapOp(qreg q, int idx1, int idx2) {
  Swap(q[idx1], q[idx2]);
}

// Controlled-Swap
__qpu__ void cSwap(qreg q, int ctrlIdx, int idx1, int idx2) {
    SwapOp::ctrl(ctrlIdx, q, idx1, idx2);
}

// Single controlled *modular* multiplication by a (mod N)
__qpu__ void cMultModN(qreg q, int ctrlIdx,
                      int a, int N,
                      int startBitIdx, int nbQubits,
                      // Auxilary qubit register
                      int auxStartBitIdx, int auxNbQubits) {
  int inv0 = 0;
  int inv1 = 1;
  int withSwap = 0;
  // QFT on the auxilary:
  int qftSize = nbQubits + 1;
  qft_range_opt_swap(q, auxStartBitIdx, qftSize, withSwap);
  for (int i = 0; i < nbQubits; ++i) {
    int tempA = 0;
    qcor::util::calcPowMultMod(tempA, i, a, N);
    int xIdx = startBitIdx + i;
    int auxBit = auxStartBitIdx + auxNbQubits - 1;
    // Doubly-controlled modular add on the Aux register.
    ccPhiAddModN(q, xIdx, ctrlIdx, auxBit, tempA, N, auxStartBitIdx, qftSize);
  }

  iqft_range_opt_swap(q, auxStartBitIdx, qftSize, withSwap);
  
  for (int i = 0; i < nbQubits; ++i) {
    int idx1 = startBitIdx + i;
    int idx2 = auxStartBitIdx + i;
    cSwap(q, ctrlIdx, idx1, idx2);
  }

  qft_range_opt_swap(q, auxStartBitIdx, qftSize, withSwap);
  
  int aInv = 0;
  qcor::util::modinv(aInv, a, N);
  for (int i = nbQubits - 1; i >= 0; --i) {
    int tempA = 0;
    qcor::util::calcPowMultMod(tempA, i, aInv, N);
    int xIdx = startBitIdx + i;
    int auxBit = auxStartBitIdx + auxNbQubits - 1;
    ccPhiAddModN_inv(q, xIdx, ctrlIdx, auxBit, tempA, N, auxStartBitIdx, qftSize);
  }

  iqft_range_opt_swap(q, auxStartBitIdx, qftSize, withSwap);
}

// Period finding:
// Input a, N 
// (1 < a < N)
// e.g. N = 15, a = 4 
// Size of qreg must be at least (4n + 2)
// where n is the number of bits which can represent N,
// i.e. N = 15 -> n = 4 (4 bits) => qreg must have at lease 18 qubits.
__qpu__ void periodFinding(qreg q, int a, int N) {
  // Down register: 0 -> (n-1): size = n
  // Up register (QFT): n -> (3n - 1): size = 2n
  // Aux register: 3n -> 4n + 1: size = n + 2
  int n = 0;
  qcor::util::calcNumBits(n, N);
  int downStart = 0;
  int downSize = n;
  int upStart = n;
  int upSize = 2*n;
  int auxStart = 3*n;
  int auxSize = n + 2;

  // Hadamard up register:
  for (int i = 0; i < upSize; ++i) {
    int bitIdx = upStart + i;
    H(q[bitIdx]);
  }

  // Init down register to 1
  X(q[downStart]);
  // Apply the multiplication gates
  for (int i = 0; i < 2*n; ++i) {
    int aTo2Toi = 0;
    qcor::util::calcExpExp(aTo2Toi, a, i, N);
    // Control bit is the upper register
    int ctrlIdx = upStart + i;
    cMultModN(q, ctrlIdx, aTo2Toi, N, downStart, downSize, auxStart, auxSize);
  }

  // Apply inverse QFT on the up register (with Swap)
  int withSwap = 1;
  iqft_range_opt_swap(q, upStart, upSize, withSwap);

  // Measure the upper register
  for (int i = 0; i < upSize; ++i) {
    int bitIdx = upStart + i;
    Measure(q[bitIdx]);
  }
  // Done: examine the shot count statistics
  // to compute the period.
}
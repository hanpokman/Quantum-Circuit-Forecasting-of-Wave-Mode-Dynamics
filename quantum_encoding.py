import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

def normalize_modes(c): # prepares raw modal coefficients so they can be represented as valid quantum state
    e = np.linalg.norm(c, axis=-1) + 1e-12 # 1 value. Length of 64-dimensional coefficients in hilbert space

    r = (c / e[..., None]).astype(np.complex64) # normalize the original vector so that the resulting norm ||r|| = 1
    return r, np.log(e).astype(np.float32)

def denormalize_modes(r, log_e): # just reverse of last function
    return (r * np.exp(log_e)[..., None]).astype(np.complex64)


def encode_state(r):
    r = np.asarray(r, np.complex128)
    n = int(np.ceil(np.log2(len(r)))) # calculates minimum amount of qubits required to store superpositions
    if len(r) < 2 ** n:
        r = np.pad(r, (0, 2 ** n - len(r))) # check if length of vector 2 is a power of 2. If not, zero pad until it is equal to 2^n

    r = r / np.linalg.norm(r)

    qc = QuantumCircuit(n)
    qc.append(StatePreparation(r), range(n)) # takes zero-padded normalized vector and calculates exact sequence of quantum gates to transform ground state |00...> to arbitrary target superposition state
    return qc
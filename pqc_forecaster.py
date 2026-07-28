import torch
import torch.nn as nn

from config import M_MODES, GRU_HIDDEN, N_QUBITS, PQC_LAYERS, SEQ_L


# isolate pairs with amplitudes that only differ at qubit q (like 010 and 000) --> apply vector to it and return statevector
def _apply_1q(state, u, q, n):
    s = state.view(-1, 2 ** q, 2, 2 ** (n - q - 1))
    return torch.einsum("bij,bajc->baic", u, s).reshape(-1, 2 ** n)


def _ry(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.complex(torch.stack([torch.stack([c, -s], -1),
                                      torch.stack([s, c], -1)], -2),
                         torch.zeros(*c.shape, 2, 2, device=c.device))


def _rz(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    z = torch.zeros_like(c)
    re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    im = torch.stack([torch.stack([-s, z], -1), torch.stack([z, s], -1)], -2)
    return torch.complex(re, im)

# ^ torch.complex takes both (real and imaginary part)
# using euler's formula e^ix = cos x + i sin x, we encode the cos part as real and sin part as imaginary

def ring_pairs(n=N_QUBITS):
    return [(q, (q + 1) % n) for q in range(n)]

def _zz_signs(n=N_QUBITS, pairs=None):
    pairs = pairs or ring_pairs(n)
    idx = torch.arange(2 ** n)
    bits = torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)])
    '''
    Basis State:    |000⟩  |001⟩  |010⟩  |011⟩  |100⟩  |101⟩  |110⟩  |111⟩
    State Index:      0      1      2      3      4      5      6      7
    ----------------------------------------------------------------------
    bits[0] (q=0):  [ 0,     0,     0,     0,     1,     1,     1,     1 ]
    bits[1] (q=1):  [ 0,     0,     1,     1,     0,     0,     1,     1 ]
    bits[2] (q=2):  [ 0,     1,     0,     1,     0,     1,     0,     1 ]
    '''

    return torch.stack([1.0 - 2.0 * (bits[a] ^ bits[b]).float()
                        for a, b in pairs])


def _cnot_perms(n=N_QUBITS, pairs=None):
    """
    CNOTs all qubits and returns reordered statis vector
    """
    pairs = pairs or ring_pairs(n)
    idx = torch.arange(2 ** n)
    perms = []
    for a, b in pairs:
        ctrl = (idx >> (n - 1 - a)) & 1
        perms.append(idx ^ (ctrl << (n - 1 - b)))
    return torch.stack(perms)

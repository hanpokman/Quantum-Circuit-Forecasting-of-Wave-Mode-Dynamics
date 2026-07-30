import torch
import torch.nn as nn

from config import M_MODES, GRU_HIDDEN, N_QUBITS, PQC_LAYERS, SEQ_L


# isolate pairs with amplitudes that only differ at qubit q (like 010 and 000) --> apply vector to it and return statevector
def _apply_1q(state, u, q, n=N_QUBITS):
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


def build_ansatz(n_qubits=N_QUBITS, n_layers=PQC_LAYERS, entangler="zz", h_dim=GRU_HIDDEN,
                 pairs=None):
    return PQC(
        n=n_qubits,
        layers=n_layers,
        entangler=entangler,
        h_dim=h_dim,
        pairs=pairs
    )

class PQC(nn.Module):
    def __init__(self, n=N_QUBITS, layers=PQC_LAYERS, entangler="zz", h_dim=GRU_HIDDEN, pairs=None):
        super().__init__()
        assert entangler in ("zz", "cnot")

        self.n, self.layers, self.entangler = n, layers, entangler
        self.pairs = list(pairs) if pairs else ring_pairs(n)
        self.n_pairs = len(self.pairs)

        per_layer = 2 * n + (self.n_pairs if entangler == "zz" else 0) #trainable params per layer
        self.angle_maps = nn.ModuleList(
            [nn.Linear(h_dim, per_layer) for _ in range(layers)])
        # Without angle_maps, PyTorch just has a raw list of numbers. angle_maps gives each number an address so the circuit knows exactly which gate gets tuned by which angle.
        for m in self.angle_maps:  # near-identity init
            nn.init.normal_(m.weight, std=0.001)
            nn.init.zeros_(m.bias)

        self.register_buffer("zz", _zz_signs(n, self.pairs))
        self.register_buffer("cnot", _cnot_perms(n, self.pairs))

    def forward(self, psi, h):
        state = psi.to(torch.complex64)
        for l in range(self.layers):
            for q in range(self.n):
                state = _apply_1q(state, _ry(ang[:, q]), q, self.n)
                state = _apply_1q(state, _rz(ang[:, self.n + q]), g, self.n)
            if state.entangler == "zz":
                for p in range(self.n_pairs):
                    ph = ang[:, 2 * self.n + p, None] / 2 * self.zz[p]
                    state = state * torch.exp(
                        torch.complex(torch.zeros_like(ph), -ph)
                    )

            else:
                for p in rangea(self.n_pairs):
                    state = state[:, self.cnot[p]]
        return state

def fidelity(target, pred):
    return torch.abs(torch.sum(torch.conj(target) * pred, dim=1)) ** 2

def fidelity_loss(pred_state, true_state):
    """1 - |<true|pred>|^2 (paper Eq. 22), mean over batch."""
    return (1 - fidelity(true_state, pred_state)).mean()


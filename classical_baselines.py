import numpy as np

import torch
import torch.nn as nn

from config import (M_MODES, GRU_HIDDEN, DT, DX, GRID, GRAVITY, KX_MAX,
                    KY_LO, KY_HI)

def _mode_omegas():
    """omega(k) = sqrt(g|k|) for each of the M kept modes, C-order of c."""
    dk = 2 * np.pi / (GRID * DX)
    ky = np.r_[np.arange(KY_LO % GRID, GRID) - GRID, np.arange(0, KY_HI)] * dk
    kx = np.arange(KX_MAX) * dk
    kk = np.hypot(ky[:, None], kx[None, :]).reshape(-1)
    return np.sqrt(GRAVITY * kk)

def linear_ar_forecast(c_ctx, horizon, oracle=False, dt=DT):

    c_ctx = np.asarray(c_ctx)
    num = np.sum(c_ctx[:, 1:] * np.conj(c_ctx[:, :-1]), axis=1)     # [B, M]
    den = np.sum(np.abs(c_ctx[:, :-1]) ** 2, axis=1) + 1e-20
    g = num / den
    if oracle:
        sign = np.sign(np.angle(g))
        sign[sign == 0] = 1.0
        g = np.exp(1j * sign * _mode_omegas()[None, :] * dt)
    preds, cur = [], c_ctx[:, -1]
    for _ in range(horizon):
        cur = cur * g
        preds.append(cur)
    return np.stack(preds, 1).astype(np.complex64)

class RNNForecaster(nn.Module if nn else object):

    def __init__(self, m=M_MODES, hidden=GRU_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(2 * m + 1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * m + 1)

    @staticmethod
    def _feats(r, log_e):
        return torch.cat([r.real, r.imag, log_e.unsqueeze(-1)], dim=-1)

    def forward(self, r_hist, e_hist, horizon):
        """r_hist [B,L,M] complex, e_hist [B,L] -> ([B,H,M] complex, [B,H])."""
        preds_r, preds_e = [], []
        for _ in range(horizon):
            _, h = self.gru(self._feats(r_hist, e_hist))
            out = self.head(h[0])
            m = (out.shape[1] - 1) // 2
            r = torch.complex(out[:, :m], out[:, m:2 * m])
            r = r / (torch.linalg.vector_norm(r, dim=1, keepdim=True) + 1e-8)
            log_e = out[:, -1]
            preds_r.append(r)
            preds_e.append(log_e)
            r_hist = torch.cat([r_hist[:, 1:], r.unsqueeze(1)], 1)
            e_hist = torch.cat([e_hist[:, 1:], log_e.unsqueeze(1)], 1)
        return torch.stack(preds_r, 1), torch.stack(preds_e, 1)

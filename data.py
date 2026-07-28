import numpy as np
import torch
from torch.utils.data import Dataset

from config import SEQ_L, HORIZON
from quantum_encoding import normalize_modes

def load_split(npz_path, split):
    z = np.load(npz_path)
    key = f"c_{split}" # if splite = "train" --> key c_train ...
    if key not in z:
        raise KeyError(f"{key} not in {npz_path} (keys: {list(z.keys())}")

    return z[key]

class SequenceData(Dataset):
    def __init__(self, npz_path, split):
        c = load_split(npz_path, split)
        self.r, self.log_e = normalize_modes(c)
        t_len = c.shape[1] # trajectory

        self.index = [(e, t) for e in range(c.shape[0])
                      for t in range(t_len - SEQ_L - HORIZON + 1)]

        def __len__(self):
            return len(self.index)

        def __getitem__(self, i):
            e, t = self.index[i]
            r = torch.from_numpy

            r = torch.from_numpy(self.r[e, t:t + SEQ_L + HORIZON])  # [L+H, M] complex
            le = torch.from_numpy(self.log_e[e, t:t + SEQ_L + HORIZON])
            return r[:SEQ_L], r[SEQ_L:], le[:SEQ_L], le[SEQ_L:]
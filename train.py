import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import BATCH, FORECAST_LR, RNN_LR, HORIZON, LAMBDA_E
from data import SequenceDataset
from pqc_forecaster import Forecaster, fidelity
from classical_baselines import RNNForecaster

DEV = "mps"

def _epoch(model, loader, opt=None):
    tot = fid = n = 0
    for r_ctx, r_fut, e_ctx, e_fut in loader:
        r_ctx, r_fut = r_ctx.to(DEV), r_fut.to(DEV)
        e_ctx, e_fut = e_ctx.to(DEV), e_fut.to(DEV)
        r_hat, e_hat = model(r_ctx, e_ctx, HORIZON)
        fids = torch.stack([fidelity(r_fut[:, k], r_hat[:, k])
                            for k in range(HORIZON)], 1)          # [B, H]
        loss = (1 - fids).sum(1).mean() + LAMBDA_E * F.mse_loss(e_hat, e_fut)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        b = len(r_ctx)
        tot += loss.item() * b
        fid += fids.mean().item() * b
        n += b
    return tot / n, fid / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pqc-zz", "pqc-cnot", "rnn"])
    ap.add_argument("--data", default="data/waves.npz")
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=BATCH)
    a = ap.parse_args()
    os.makedirs(a.ckpt, exist_ok=True)

    train_loader = DataLoader(SequenceDataset(a.data, "train"), a.batch, shuffle=True)
    val_loader = DataLoader(SequenceDataset(a.data, "val"), a.batch)

    if a.stage == "rnn":
        model, lr = RNNForecaster().to(DEV), RNN_LR
    else:
        model = Forecaster(entangler=a.stage.split("-")[1]).to(DEV)
        lr = FORECAST_LR
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[{a.stage}] {sum(p.numel() for p in model.parameters())} parameters")

    best = -1.0
    path = os.path.join(a.ckpt, f"{a.stage}.pt")
    for ep in range(a.epochs):
        model.train()
        tr_loss, tr_fid = _epoch(model, train_loader, opt)
        model.eval()
        with torch.no_grad():
            _, va_fid = _epoch(model, val_loader)
        print(f"[{a.stage}] epoch {ep + 1}/{a.epochs}  loss {tr_loss:.5f}  "
              f"train-fid {tr_fid:.4f}  val-fid {va_fid:.4f}")
        if va_fid > best:
            best = va_fid
            torch.save(model.state_dict(), path)
    print(f"[{a.stage}] best val fidelity {best:.4f} -> {path}")


if __name__ == "__main__":
    main()

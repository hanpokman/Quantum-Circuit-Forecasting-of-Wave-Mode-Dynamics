import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import M_MODES, SEQ_L, HORIZON, GRID, DT, DX, GRAVITY, KX_MAX, KY_LO, KY_HI
from data import SequenceDataset
from generate_data import simulate_wave_field, extract_fourier_modes, modes_to_field
from quantum_encoding import normalize_modes, denormalize_modes
from evaluate import np_fidelity, field_rmse, evaluate_forecaster

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _skip_no_torch():
    if not HAVE_TORCH:
        pytest.skip("torch not installed")


# ------------------------- U12: split / window integrity --------------------
def _make_synthetic_npz(path, n_train=160, n_val=20, n_test=20, T=64, m=M_MODES,
                        seed=0):
    rng = np.random.default_rng(seed)

    def make(n):
        return (rng.standard_normal((n, T, m))
                + 1j * rng.standard_normal((n, T, m))).astype(np.complex64)

    np.savez(path, c_train=make(n_train), c_val=make(n_val), c_test=make(n_test))


def test_episode_splits_are_disjoint_and_windows_do_not_cross_episodes(tmp_path):
    npz_path = tmp_path / "synthetic.npz"
    _make_synthetic_npz(npz_path)
    z = np.load(npz_path)

    hashes = set()
    for split in ("train", "val", "test"):
        c = z[f"c_{split}"]
        for e in range(c.shape[0]):
            h = c[e].tobytes()
            assert h not in hashes, "episode reused across splits"
            hashes.add(h)
    assert z["c_train"].shape[0] == 160
    assert z["c_val"].shape[0] == 20
    assert z["c_test"].shape[0] == 20
    assert sum(z[f"c_{s}"].shape[0] for s in ("train", "val", "test")) == 200

    windows_per_ep = z["c_train"].shape[1] - SEQ_L - HORIZON + 1
    assert windows_per_ep == 52                       # 64 - 8 - 5 + 1

    ds = SequenceDataset(str(npz_path), "train")
    assert len(ds) == 160 * 52
    t_max = z["c_train"].shape[1] - SEQ_L - HORIZON
    for e, t in ds.index:                              # windows never cross episodes
        assert 0 <= e < 160
        assert 0 <= t <= t_max


# ------------------------- U13: dataset metadata / shape --------------------
def test_generated_dataset_matches_configuration():
    eta, _ = simulate_wave_field(2, seed=0)
    c = extract_fourier_modes(eta)
    assert c.shape[-2:] == (eta.shape[1], M_MODES) == (64, 64)
    assert c.dtype == np.complex64
    assert np.isfinite(c.real).all() and np.isfinite(c.imag).all()
    assert (GRID, M_MODES, DT, DX, GRAVITY) == (64, 64, 0.1, 1.0, 9.81)
    assert (KX_MAX, KY_LO, KY_HI) == (8, -3, 5)


# ------------------------- U14: evaluation-metric correctness ---------------
def test_evaluation_metrics_on_known_predictions():
    rng = np.random.default_rng(0)
    c = (rng.standard_normal((3, M_MODES))
         + 1j * rng.standard_normal((3, M_MODES))).astype(np.complex64)
    r, _ = normalize_modes(c)

    assert np.allclose(np_fidelity(r, r), 1.0, atol=1e-6)          # pred == target
    assert field_rmse(c, c) == 0.0

    r0 = r[0]                                                       # orthogonal state
    w = rng.standard_normal(M_MODES) + 1j * rng.standard_normal(M_MODES)
    w = w - np.vdot(r0, w) * r0
    w = (w / np.linalg.norm(w)).astype(np.complex64)
    assert np_fidelity(r0[None], w[None])[0] < 1e-6

    delta = (c[0] + 0.01).astype(np.complex64)                      # modal perturbation
    val = field_rmse(delta[None], c[0][None])
    assert np.isfinite(val) and val > 0

    c_hat = (rng.standard_normal((5, HORIZON, M_MODES))
             + 1j * rng.standard_normal((5, HORIZON, M_MODES))).astype(np.complex64)
    c_true = (rng.standard_normal((5, HORIZON, M_MODES))
              + 1j * rng.standard_normal((5, HORIZON, M_MODES))).astype(np.complex64)
    out = evaluate_forecaster(c_hat, c_true)
    for k in range(1, HORIZON + 1):
        f, r_ = out[f"fid@{k}"], out[f"fieldRMSE@{k}"]
        assert np.isfinite(f) and 0 <= f <= 1
        assert np.isfinite(r_) and r_ >= 0


# ------------------------- U15: global-phase consistency --------------------
def test_global_phase_is_not_invariant_in_reconstructed_wave_field():
    eta, _ = simulate_wave_field(1, seed=9)
    c = extract_fourier_modes(eta)[0, 0]
    r, log_e = normalize_modes(c)
    r_phi = (np.exp(1j * np.pi / 2) * r).astype(np.complex64)

    fid = np_fidelity(r[None], r_phi[None])[0]
    assert abs(fid - 1.0) < 1e-6                                    # state fidelity ~1

    field = modes_to_field(denormalize_modes(r, log_e))
    field_phi = modes_to_field(denormalize_modes(r_phi, log_e))
    rmse = float(np.sqrt(np.mean((field - field_phi) ** 2)))
    assert rmse > 0                                                 # physical field differs


# ------------------------- U16: checkpoint metadata round trip --------------
def test_checkpoint_reconstructs_exact_model_architecture(tmp_path):
    _skip_no_torch()
    from pqc_forecaster import Forecaster

    model = Forecaster(entangler="cnot", n_layers=1)               # nondefault arch
    ckpt = {"state_dict": model.state_dict(), "entangler": "cnot", "n_layers": 1}
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    loaded = torch.load(path, map_location="cpu")
    model2 = Forecaster(entangler=loaded["entangler"], n_layers=loaded["n_layers"])
    model2.load_state_dict(loaded["state_dict"])

    r = torch.randn(2, SEQ_L, M_MODES, dtype=torch.complex64)
    r = r / torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    e = torch.randn(2, SEQ_L)
    model.eval(); model2.eval()
    with torch.no_grad():
        r1, e1 = model(r, e, HORIZON)
        r2, e2 = model2(r, e, HORIZON)
    assert torch.allclose(r1, r2, atol=1e-6) and torch.allclose(e1, e2, atol=1e-6)
    # NOTE: train.py/evaluate.py currently save/load a bare state_dict, not
    # this structured {state_dict, entangler, n_layers, ...} form -- this
    # test documents the round-trip the proposal recommends adopting there.


# ------------------------- U17: end-to-end smoke integration ----------------
@pytest.mark.integration
def test_end_to_end_smoke_pipeline(tmp_path):
    import subprocess
    import sys as _sys
    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = tmp_path / "waves_smoke.npz"
    ckpt = tmp_path / "ckpt_smoke"
    out = tmp_path / "table_smoke.csv"
    subprocess.run(
        [_sys.executable, os.path.join(root, "run_all.py"),
         "--episodes", "10", "--epochs", "1",
         "--data", str(data), "--ckpt", str(ckpt), "--out", str(out)],
        check=True, cwd=root)

    assert data.exists() and out.exists()
    for stage in ("pqc-zz", "pqc-cnot", "rnn"):
        assert (ckpt / f"{stage}.pt").exists()
    df = pd.read_csv(out, index_col=0)
    for k in range(1, HORIZON + 1):
        assert f"fid@{k}" in df.columns and f"fieldRMSE@{k}" in df.columns
    assert np.isfinite(df.values.astype(float)).all()
    # smoke values only -- not a scientific result (5 episodes, 1 epoch)


# ------------------------- U18: reproducibility ------------------------------
def test_fixed_seed_reproduces_data_and_model_initialization():
    _skip_no_torch()
    eta1, _ = simulate_wave_field(2, seed=42)
    eta2, _ = simulate_wave_field(2, seed=42)
    assert np.array_equal(eta1, eta2)
    assert np.array_equal(extract_fourier_modes(eta1), extract_fourier_modes(eta2))

    from pqc_forecaster import Forecaster
    torch.manual_seed(123)
    m1 = Forecaster("zz")
    torch.manual_seed(123)
    m2 = Forecaster("zz")
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
    # NOTE: generate_data.py --seed is already fully deterministic (checked
    # above). train.py/evaluate.py/run_all.py do not yet expose a --seed CLI
    # flag that calls torch.manual_seed -- that CLI plumbing is still a gap.


# ------------------------- U19: parameter-matching ---------------------------
def test_classical_baseline_parameter_count_is_matched():
    _skip_no_torch()
    from pqc_forecaster import Forecaster
    from classical_baselines import RNNForecaster

    p_zz = sum(p.numel() for p in Forecaster("zz").parameters())
    p_rnn64 = sum(p.numel() for p in RNNForecaster(hidden=64).parameters())
    p_rnn58 = sum(p.numel() for p in RNNForecaster(hidden=58).parameters())

    assert abs(p_rnn64 - p_zz) / p_zz > 0.05      # current default (64): NOT matched
    assert abs(p_rnn58 - p_zz) / p_zz <= 0.05     # hidden=58: parameter-matched


# ------------------------- U20: Torch<->Qiskit equivalence (optional) -------
def _bitrev_perm(n):
    idx = np.arange(2 ** n)
    rev = np.zeros_like(idx)
    for q in range(n):
        rev |= ((idx >> q) & 1) << (n - 1 - q)
    return rev


@pytest.mark.qiskit
def test_torch_and_qiskit_statevectors_are_equivalent():
    pytest.importorskip("qiskit")
    _skip_no_torch()
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import Statevector
    from pqc_forecaster import PQC

    rng = np.random.default_rng(0)
    for entangler in ("zz", "cnot"):
        pqc = PQC(entangler=entangler, layers=1)
        n = pqc.n
        psi0 = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
        psi0 = (psi0 / np.linalg.norm(psi0)).astype(np.complex64)
        h = rng.standard_normal(64).astype(np.float32)
        with torch.no_grad():
            out_torch = pqc(torch.from_numpy(psi0)[None],
                            torch.from_numpy(h)[None])[0].numpy()
            ang = pqc.angle_maps[0](torch.from_numpy(h)[None])[0].numpy()

        perm = _bitrev_perm(n)                    # torch (MSB-first) <-> qiskit (LSB-first)
        qc = QuantumCircuit(n)
        qc.append(StatePreparation(psi0[perm]), range(n))
        for q in range(n):
            qc.ry(float(ang[q]), q)
            qc.rz(float(ang[n + q]), q)
        if entangler == "zz":
            for p, (a, b) in enumerate(pqc.pairs):
                qc.rzz(float(ang[2 * n + p]), a, b)
        else:
            for a, b in pqc.pairs:
                qc.cx(a, b)

        out_qiskit = Statevector(qc).data[perm]
        fid = abs(np.vdot(out_torch, out_qiskit)) ** 2
        assert fid > 1 - 1e-4, entangler

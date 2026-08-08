# Test suite

Two kinds of check, kept separate: **software correctness tests** (this file)
verify the code implements the intended math; **scientific experiments**
(`evaluate.py`, `run_all.py`, proposal sec IV.C-D) measure forecasting
performance. Correctness tests never appear as rows in a results table.

## Files
- `tests/test_core.py` — U1-U11, pre-existing.
- `tests/test_additional.py` — U12-U20, added here.

## What's covered

| ID | Test | Checks |
|----|------|--------|
| U1 | `test_fields_real_finite_nondegenerate` | simulated fields real/finite/non-trivial |
| U2 | `test_mode_extraction_roundtrip` | FFT-mode extraction ↔ field reconstruction |
| U3 | `test_band_captures_most_variance` | retained 64-mode band explains >85% variance |
| U4 | `test_reconstruction_is_real_field` | Hermitian completion → real float32 field |
| U5 | `test_normalize_roundtrip_and_unit_norm` | amplitude (de)normalization |
| U6 | `test_linear_ar_near_exact_on_linear_dynamics` | Linear-AR solves clean simulator |
| U7 | `test_dispersion_oracle_matches_data` | fitted phase matches deep-water dispersion |
| U8 | `test_pqc_preserves_norm_both_entanglers` | PQC unitarity (ZZ, CNOT) |
| U9 | `test_pqc_near_identity_init_zz` | ZZ circuit starts near identity |
| U10 | `test_gates_match_explicit_unitaries` | RY/ZZ/CNOT vs explicit matrices |
| U11 | `test_rollout_shapes_and_grad` | 5-step rollout shapes + gradient flow |
| U12 | `test_episode_splits_are_disjoint_and_windows_do_not_cross_episodes` | episode-disjoint splits; 52 windows/episode; 160/20/20 split sizes |
| U13 | `test_generated_dataset_matches_configuration` | generated shapes/dtype match `config.py` |
| U14 | `test_evaluation_metrics_on_known_predictions` | `np_fidelity`/`field_rmse`/`evaluate_forecaster` on known cases |
| U15 | `test_global_phase_is_not_invariant_in_reconstructed_wave_field` | global phase ≈1 state fidelity but changes the physical field |
| U16 | `test_checkpoint_reconstructs_exact_model_architecture` | structured checkpoint round-trip reproduces outputs exactly |
| U17 | `test_end_to_end_smoke_pipeline` (`@integration`) | generate→train→checkpoint→eval→CSV pipeline runs end to end |
| U18 | `test_fixed_seed_reproduces_data_and_model_initialization` | same seed ⇒ same data / same model init |
| U19 | `test_classical_baseline_parameter_count_is_matched` | RNN(64) is *not* param-matched to PQC-ZZ; RNN(58) is |
| U20 | `test_torch_and_qiskit_statevectors_are_equivalent` (`@qiskit`) | Torch statevector sim ≡ gate-for-gate Qiskit transcription |

## Running

```bash
python -m pytest -q tests                       # U1-U16, U18-U19 (fast, default)
python -m pytest -q -m integration               # U17 (spawns a real training run)
python -m pytest -q -m qiskit                    # U20 (needs `pip install qiskit`)
python -m pytest -q                              # everything available
```

## Interpreting results

- **Pass** = the implementation matches the intended math for that unit; it is
  *not* evidence the trained model forecasts well (that's what the
  experiments in sec IV.C-D are for).
- U19 is a *diagnostic*: it's designed so the 64-unit RNN assertion documents
  the current ~12% mismatch, and the 58-unit assertion documents the fix.
  Both are expected to hold simultaneously.
- U16 and U18 exercise round-trip/reproducibility logic directly (structured
  checkpoint dict, `torch.manual_seed`) rather than the CLI. `train.py`,
  `evaluate.py`, and `run_all.py` do not yet save structured checkpoints or
  accept a `--seed` flag — that CLI/checkpoint-format work from the proposal
  is still open; these tests will keep passing once it lands, since they
  test the underlying mechanism, not the flags.
- U17 is a smoke test (10 episodes, 1 epoch) — its numbers are not
  publishable results, only a check that every stage produces valid,
  finite output.
- U20 checks simulator equivalence, not hardware noise; skip silently
  without Qiskit installed.

# --- dataset / simulation -------------------------------------------------
GRID = 64               # height-field sample grid, N = GRID*GRID points
DX = 1.0                # metres between adjacent sample points
EP_LEN = 64             # frames per episode
DT = 0.1                # seconds between frames
GRAVITY = 9.81

WIND_SPEED_RANGE = (4.0, 12.0)      # m/s, drawn uniformly per episode
WIND_DIR_RANGE = (0.0, 6.283185)    # rad, drawn uniformly per episode

# --- modal representation --------------------------------------------------
M_MODES = 64            # M complex modes -> log2(M) = 6 qubits
KX_MAX = 8              # kept wavenumber indices: kx in [0,8), ky in [-3,5)
KY_LO, KY_HI = -3, 5    # half-plane low-pass (quadrant-only would drop modes)

# --- temporal forecasting ---------------------------------------------------
SEQ_L = 8               # context frames L (paper)
HORIZON = 5             # future frames H (paper)
GRU_HIDDEN = 64         # conditioning vector h_t dimensionality
N_QUBITS = 6            # log2(M_MODES)
PQC_LAYERS = 3          # data re-uploading layers
LAMBDA_E = 0.1          # weight of log-energy head loss
FORECAST_LR = 5e-3
RNN_LR = 1e-3           # classical GRU baseline
BATCH = 32

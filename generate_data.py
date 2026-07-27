import argparse
import os
import numpy as np

from config import (GRID, DX, EP_LEN, DT, GRAVITY, M_MODES, KX_MAX, KY_LO,
                    KY_HI, WIND_SPEED_RANGE, WIND_DIR_RANGE)

class SpectralOcean:
    def __init__(self,
                 wind_speed,
                 wind_dir_rad,
                 rng,
                 n=GRID,
                 domain=GRID * DX,
                 spectrum="pm"):

        self.n = n

        k1 = 2 * np.pi * np.fft.fftfreq(n, d=domain / n)
        kx, ky = np.meshgrid(k1, k1)  # rows = ky, cols = kx
        k = np.hypot(kx, ky)
        k_safe = np.where(k == 0, 1e-9, k)
        omega = np.sqrt(GRAVITY * k_safe)

        alpha = 8.1e-3 # phillips constant
        wp = 0.877 * GRAVITY ** 2 / max(wind_speed, 1e-6) # peak angular frequency --> the frequency where ocean wave energy is peak
        s_omega = alpha * GRAVITY ** 2 / (omega ** 5) * np.exp(-1.25 * (wp/omega) ** 4)

        if spectrum == "jonswap":
            gamma = 3.3
            sig = np.where(omega <= wp, 0.07, 0.09)
            s_omega *= gamma ** np.exp(-((omega - wp) ** 2) / (2 * sig ** 2 * wp ** 2))

        jac = GRAVITY / (2 * omega)
        psi = s_omega * jac / k_safe # jacobian transformation formula -- translator that converts wave energy from time-speed format in s(w) to spatial map
        '''
        Jacobian transformation derivation
        
        In 1D, total energy across interval dw is 
        
        E = S(w) dw, where S(w) gives the energy by frequency of a wave
        
        In 2D polar coordinates, the energy element across an area k dk dtheta is 
        
        E = psi(k, theta) k dk d theta = S(w) dw D(theta) # D(thetas) is a direction spreading function
        
        psi(k, theta) = S(w) * dw/dk * 1/k * D(theta)
        
        Since we know w = sqrt(gk) given by the deep-water dispersion relation
        dw/dk = g/[2*sqrt(gk)]
        
        dw/dk = g/2w (since sqrt(gk) = w)

        Substitute everything back and we get the psi formula
        '''


        # cosine^2 spreading basically it's a mask that applies to original radial symmetry wave to make it go in a certain angle
        dtheta = np.arctan2(ky, kx) - wind_dir_rad
        spread = np.where(np.cos(dtheta) > 0, (2 / np.pi) * np.cos(dtheta) ** 2, 0.0)
        psi *= spread
        psi[k == 0] = 0.0

        dk = (2 * np.pi / domain) ** 2 # physical surface area of a single square grid cell
        xi = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)

        self.h0 = xi * np.sqrt(psi * dk) # create wave amplitudes at t = 0 using Tessendorf (1999)
        self.h0_conj_neg = np.conj(np.roll(np.flip(self.h0), shift=(1, 1), axis=(0, 1))) # creates the exact conjugate pair needed so that when the ocean propagates forward in time, the spatial surface elevation stays 100% real-valued.
        self.omega = np.sqrt(GRAVITY * k) # redefine


    def height(self, t):
        """eta [n, n] float32 at time t (real by Hermitian construction)."""
        ph = np.exp(1j * self.omega * t)
        hk = self.h0 * ph + self.h0_conj_neg * np.conj(ph)
        return np.real(np.fft.ifft2(hk)).astype(np.float32) * self.n * self.n # convert to 3D


def simulate_wave_field(n_episodes, # uses SpectralOcean object to generate different ocean simulations
                        grid_size=GRID,
                        n_timesteps=EP_LEN,
                        dt=DT,
                        seed=None,
                        wind_speed_range=WIND_SPEED_RANGE,
                        wind_dir_range=WIND_DIR_RANGE,
                        spectrum="pm"):
    rng = np.random.default_rng(seed)
    eta = np.empty((n_episodes, n_timesteps, grid_size, grid_size), np.float32)
    winds = np.empty((n_episodes, 2), np.float32)
    for e in range(n_episodes):
        u = rng.uniform(*wind_speed_range)
        th = rng.uniform(*wind_dir_range)
        ocean = SpectralOcean(u, th, rng, n=grid_size, spectrum=spectrum)
        t0 = rng.uniform(0, 100)
        for t in range(n_timesteps):
            eta[e, t] = ocean.height(t0 + t * dt)
        winds[e] = u, th
    return eta, winds


def _kept_indices(grid=GRID):# returns which 64 modes to keep
    """(rows, cols) index arrays of the kept half-plane band, C-order of c."""
    rows = np.r_[KY_LO % grid:grid, 0:KY_HI]      # ky = -3..-1, 0..4
    cols = np.arange(KX_MAX)                      # kx = 0..7
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.reshape(-1), cc.reshape(-1)


def extract_fourier_modes(eta, m=M_MODES): # extracts the 64 modes
    """ Wave --> Modes """
    grid = eta.shape[-1]
    f = np.fft.fft2(eta) / (grid * grid)
    rr, cc = _kept_indices(grid)
    c = f[..., rr, cc]
    assert c.shape[-1] == m
    return c.astype(np.complex64)

def modes_to_field(c, grid=GRID):
    """

    the reverse. 64 numbers in → wave picture out (4096 numbers).
    """
    rr, cc = _kept_indices(grid)
    f = np.zeros(c.shape[:-1] + (grid, grid), np.complex64)
    f[..., rr, cc] = c
    kept = set(zip(rr.tolist(), cc.tolist()))
    for r, col in kept:
        mr, mc = (-r) % grid, (-col) % grid
        if (mr, mc) not in kept:
            f[..., mr, mc] = np.conj(f[..., r, col])
    return (np.real(np.fft.ifft2(f)) * grid * grid).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/waves.npz")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wind-speed", type=float, nargs=2, default=WIND_SPEED_RANGE,
                    help="override wind speed range (generalization check)")
    ap.add_argument("--spectrum", default="pm", choices=["pm", "jonswap"],
                    help="wave spectrum family (jonswap: fetch-limited, "
                         "peakier — another distribution-shift test set)")

    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    rng = np.random.default_rng(a.seed)
    n_val = max(1, int(a.episodes * a.val_frac))
    n_test = max(1, int(a.episodes * a.test_frac))
    n_train = a.episodes - n_val - n_test

    c_all = np.empty((a.episodes, EP_LEN, M_MODES), np.complex64)
    winds = np.empty((a.episodes, 2), np.float32)
    for e in range(a.episodes):  # stream episodes to bound memory
        eta, w = simulate_wave_field(1, seed=rng.integers(2 ** 31),
                                     wind_speed_range=tuple(a.wind_speed),
                                     spectrum=a.spectrum)
        c_all[e] = extract_fourier_modes(eta[0])
        winds[e] = w[0]
        if (e + 1) % 20 == 0:
            print(f"episode {e + 1}/{a.episodes}")

    perm = rng.permutation(a.episodes)  # split BY EPISODE, never by frame
    tr, va, te = np.split(perm, [n_train, n_train + n_val])
    np.savez_compressed(
        a.out,
        c_train=c_all[tr], c_val=c_all[va], c_test=c_all[te],
        winds_train=winds[tr], winds_val=winds[va], winds_test=winds[te],
        grid_size=GRID, M=M_MODES, dt=DT, dx=DX, gravity=GRAVITY,
        kx_max=KX_MAX, ky_lo=KY_LO, ky_hi=KY_HI)
    print(f"wrote {a.out}: train {n_train} / val {n_val} / test {n_test} episodes, "
          f"T={EP_LEN}, M={M_MODES}")


if __name__ == "__main__":
    main()

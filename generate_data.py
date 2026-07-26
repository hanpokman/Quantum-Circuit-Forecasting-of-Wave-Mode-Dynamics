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


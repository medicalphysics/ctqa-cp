# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:22:01 2015

@author: erlean
"""
from __future__ import division
import numpy as np


def nufftfreqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins"""
    return df * np.arange(-(M // 2), M - (M // 2))


def _compute_grid_params(M, eps):
    # Choose Msp & tau from eps following Dutt & Rokhlin (1993)
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy "
                         "1e-33 < eps < 1e-1.".format(eps))
    ratio = 2 if eps > 1E-11 else 3
    Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
    Mr = max(ratio * M, 2 * Msp)
    lambda_ = Msp / (ratio * (ratio - 0.5))
    tau = np.pi * lambda_ / M ** 2
    return Msp, Mr, tau


def nufft(x, y, M, df=1.0, iflag=1, eps=1E-15):
    """Fast Non-Uniform Fourier Transform"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)

    # Construct the convolved grid ftau:
    # this replaces the loop used above
    ftau = np.zeros(Mr, dtype=y.dtype)
    hx = 2 * np.pi / Mr
    xmod = (x * df) % (2 * np.pi)
    m = 1 + (xmod // hx).astype(int)
    mm = np.arange(-Msp, Msp)
    mpmm = m + mm[:, np.newaxis]
    spread = y * np.exp(-0.25 * (xmod - hx * mpmm) ** 2 / tau)
    np.add.at(ftau, mpmm % Mr, spread)

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])

    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau

if __name__ == '__main__':
    a = np.random.random_sample(size=80)
    x = np.sin(np.linspace(0,np.pi*2, 80))

    M = 40

    print len(nufft(x, a, M))

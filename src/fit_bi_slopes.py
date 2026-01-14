# scr/fit_bi_slopes.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat

from utils import load_csv, save_figure
from constants import TESLAMETER_U_B_MT


@dataclass
class SlopeResult:
    label: str
    a_T_per_A: float       # best-fit slope [T/A]
    U_a_ext_T_per_A: float # uncertainty from pentes extrêmes [T/A]
    sigma_a_T_per_A: float # regression-based std dev [T/A]
    I_max_A: float         # max current used [A]


def fit_slope_through_origin(
    df: pd.DataFrame,
    label: str,
    u_B_mT: float = TESLAMETER_U_B_MT,
) -> SlopeResult:
    """
    Fit B = a I through the origin, using:
    - least squares for a,
    - pentes extrêmes (vertical uncertainty on B) for U(a).

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'I_A' and 'B_mT'.
    label : str
        Human-readable label for printing/logging.
    u_B_mT : float
        Instrumental uncertainty on B, in mT.

    Returns
    -------
    SlopeResult
        a (T/A), U(a) from pentes extrêmes, sigma_a from regression,
        and I_max for reference.
    """
    if not {"I_A", "B_mT"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'I_A' and 'B_mT' columns.")

    # Convert to numpy arrays and SI units
    I = df["I_A"].to_numpy(dtype=float)            # [A]
    B_T = df["B_mT"].to_numpy(dtype=float) * 1e-3  # [T]

    # --- 1) Least-squares fit through origin: a_fit = sum(I*B)/sum(I^2)
    sum_I2 = np.sum(I**2)
    a_fit = np.sum(I * B_T) / sum_I2  # [T/A]

    # Regression-based std dev on a (for information only)
    residuals = B_T - a_fit * I
    N = len(I)
    if N > 1:
        sigma_B = np.sqrt(np.sum(residuals**2) / (N - 1))
        sigma_a = sigma_B / np.sqrt(sum_I2)
    else:
        sigma_a = np.nan

    # --- 2) Pentes extrêmes (vertical uncertainty on B only)
    I_max = float(I.max())
    B_fit_max = a_fit * I_max  # [T]

    u_B_T = u_B_mT * 1e-3  # mT -> T

    a_min = (B_fit_max - u_B_T) / I_max
    a_max = (B_fit_max + u_B_T) / I_max
    U_a_ext = (a_max - a_min) / 2.0  # [T/A]

    return SlopeResult(
        label=label,
        a_T_per_A=float(a_fit),
        U_a_ext_T_per_A=float(U_a_ext),
        sigma_a_T_per_A=float(sigma_a),
        I_max_A=I_max,
    )


def plot_bi_calibration(
    df: pd.DataFrame,
    res: SlopeResult,
    filename_stem: str,
    u_B_mT: float = TESLAMETER_U_B_MT,
) -> None:
    """
    Plot B(I) with error bars and fitted line.

    Parameters
    ----------
    df : DataFrame
        Must contain 'I_A' and 'B_mT'.
    res : SlopeResult
        Result of the slope fit (a in T/A).
    filename_stem : str
        Stem used for saving the figure (e.g. 'BI_grandes_mu0').
    u_B_mT : float
        Vertical uncertainty on B, in mT.
    """
    I = df["I_A"].to_numpy(dtype=float)
    B_mT = df["B_mT"].to_numpy(dtype=float)

    # Fitted line in mT: B_fit = (a_T_per_A * 1e3) * I
    a_mT_per_A = res.a_T_per_A * 1e3

    I_fit = np.linspace(0.0, I.max() * 1.05, 200)
    B_fit_mT = a_mT_per_A * I_fit

    fig, ax = plt.subplots()

    # Experimental points with error bars
    ax.errorbar(
        I,
        B_mT,
        yerr=u_B_mT,
        fmt="o",
        markersize=3,
        capsize=2,
        label="Mesures",
    )

    # Fitted line
    ax.plot(
        I_fit,
        B_fit_mT,
        label="Ajustement linéaire $B = aI$",
    )

    ax.set_xlabel("Courant I (A)")
    ax.set_ylabel("Champ B (mT)")
    ax.set_title(res.label)

    ax.legend()
    fig.tight_layout()

    save_figure(fig, filename_stem)


def main() -> None:
    configs = [
        ("etalonage_grande_bobine_bz.csv",
         "Grandes bobines (μ0)",
         "BI_grandes_mu0"),
        ("etalonage_grande_bobine_thomson.csv",
         "Grandes bobines (Thomson)",
         "BI_grandes_thomson"),
        ("etalonage_petite_bobine_thomson.csv",
         "Petites bobines (Thomson)",
         "BI_petites_thomson"),
    ]

    print("=== Slopes B(I) from calibration datasets ===\n")

    for filename, label, fig_stem in configs:
        df = load_csv(filename)
        res = fit_slope_through_origin(df, label)

        # Slope as an uncertainties object (for nice printing / later reuse)
        a_u = ufloat(res.a_T_per_A, res.U_a_ext_T_per_A)

        a_mT_per_A = res.a_T_per_A * 1e3
        U_a_mT_per_A = res.U_a_ext_T_per_A * 1e3

        print(f"{label}:")
        print(f"  I_max          = {res.I_max_A:.3f} A")
        print(f"  a_fit          = {a_mT_per_A:.3f} mT/A")
        print(f"  U(a) (ext.)    = {U_a_mT_per_A:.3f} mT/A")
        print(f"  a ± U(a)       = ({a_u.nominal_value*1e3:.3f} ± "
              f"{a_u.std_dev*1e3:.3f}) mT/A")
        print(f"  sigma_a (LSQ)  = {res.sigma_a_T_per_A*1e3:.3e} mT/A")
        print()

        # Plot and save figure
        plot_bi_calibration(df, res, fig_stem)


if __name__ == "__main__":
    main()

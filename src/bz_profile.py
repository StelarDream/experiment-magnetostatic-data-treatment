# scr/bz_profile.py
from __future__ import annotations

from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_csv, save_figure
from constants import TESLAMETER_U_B_MTESLA, POSITION_U_Z_MM


def find_plateau_bounds(
    df: pd.DataFrame,
    frac: float = 0.01,
) -> Tuple[float, float]:
    """
    Find the z positions delimiting the plateau region around B_max.

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'z_mm' and 'B_mT'.
    frac : float
        Relative tolerance on B around B_max (e.g. 0.01 for ±1%).

    Returns
    -------
    z_left_mm : float
        Left boundary of the plateau (mm).
    z_right_mm : float
        Right boundary of the plateau (mm).
    """
    if not {"z_mm", "B_mT"}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns 'z_mm' and 'B_mT'.")

    df_sorted = df.sort_values("z_mm").reset_index(drop=True)

    B_max = df_sorted["B_mT"].max()
    idx_max = df_sorted["B_mT"].idxmax()

    # Tolerance: points kept if B >= (1 - frac) * B_max
    threshold = B_max * (1.0 - frac)

    left = idx_max
    while left - 1 >= 0 and df_sorted.loc[left - 1, "B_mT"] >= threshold:
        left -= 1

    right = idx_max
    last_index = len(df_sorted) - 1
    while right + 1 <= last_index and df_sorted.loc[right + 1, "B_mT"] >= threshold:
        right += 1

    z_left_mm = float(df_sorted.loc[left, "z_mm"])
    z_right_mm = float(df_sorted.loc[right, "z_mm"])
    return z_left_mm, z_right_mm


def main() -> None:
    # 1) Load raw data
    df = load_csv("champ_magnetique_distance_bz.csv")

    # 2) Plateau bounds (±1% around B_max)
    z_left_mm, z_right_mm = find_plateau_bounds(df, frac=0.01)

    # 3) Plateau stats
    df_sorted = df.sort_values("z_mm").reset_index(drop=True)
    mask_plateau = (df_sorted["z_mm"] >= z_left_mm) & (df_sorted["z_mm"] <= z_right_mm)
    plateau_df = df_sorted[mask_plateau].copy()

    B_mean_mT = plateau_df["B_mT"].mean()
    B_std_mT = plateau_df["B_mT"].std(ddof=1)
    z_center_mm = plateau_df["z_mm"].mean()

    print("=== Plateau (±1%) ===")
    print(f"z_left  = {z_left_mm:.1f} mm")
    print(f"z_right = {z_right_mm:.1f} mm")
    print(f"z_centre plateau = {z_center_mm:.1f} mm")
    print(f"B_moyen = {B_mean_mT:.3f} mT")
    print(f"σ(B)    = {B_std_mT:.3f} mT")
    print(f"U(B)    = {TESLAMETER_U_B_MTESLA:.3f} mT")
    print(f"U(z)    = {POSITION_U_Z_MM:.2f} mm")

    # 4) Plot
    fig, ax = plt.subplots()

    # Markers: keep default look (just error bars + 'o')
    ax.errorbar(
        df_sorted["z_mm"],
        df_sorted["B_mT"],
        yerr=TESLAMETER_U_B_MTESLA,
        fmt="o",
        markersize=3,
        capsize=2,
        label="Mesures B(z)",
    )

    # Semi-transparent region for plateau
    plateau_label = "Zone de champ quasi uniforme (±1%)"
    ax.axvspan(
        z_left_mm,
        z_right_mm,
        alpha=0.15,
        color="tab:blue",
        label=plateau_label,
    )

    # Dashed blue lines at the boundaries
    ax.axvline(z_left_mm, color="tab:blue", linestyle="--")
    ax.axvline(z_right_mm, color="tab:blue", linestyle="--")

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("B (mT)")
    ax.set_title("Profil axial du champ magnétique B(z) — grandes bobines")

    ax.legend()
    fig.tight_layout()
    save_figure(fig, "bz_profile_plateau_region")

if __name__ == "__main__":
    main()

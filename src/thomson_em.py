from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from utils import load_csv, save_figure
from constants import (
    TESLAMETER_U_B_MT,
)

# === Experimental constants ====================================================

# Slopes from your calibration (convert mT/A -> T/A)
A_SMALL_T_per_A = 3.934e-3
U_A_SMALL_T_per_A = 0.033e-3

A_LARGE_T_per_A = 0.659e-3
U_A_LARGE_T_per_A = 0.015e-3


# === Correct geometry radius ====================================================
def radius_from_deflection(Df_cm: float) -> float:
    """
    Correct circular-trajectory radius for your Thomson geometry.

    Beam exits at (0,0) with horizontal tangent;
    Observed at (Df_cm, -1 cm).

    => Circle center is BELOW the beam.
    Equation of radius in centimeters:
        R_cm = (Df^2 + 1) / 2

    Convert to meters before returning.
    """
    R_cm = (Df_cm**2 + 1.0) / 2.0  # cm
    return R_cm / 100.0            # convert to meters


# === e/m formula ===============================================================

def compute_em(Ue_V: float, B_T: float, Re_m: float) -> float:
    """Compute e/m = 2*Ue / (B^2 * Re^2)."""
    return 2 * Ue_V / (B_T**2 * Re_m**2)


# === Data structures ===========================================================

@dataclass
class EMResult:
    df_row: pd.Series
    Re_m: float
    B_T: float
    em_value: float


# === Processing logic ==========================================================

def process_dataset(df: pd.DataFrame, a_T_per_A: float) -> list[EMResult]:
    results = []

    for _, row in df.iterrows():
        # Skip invalid rows (Df=0 or I=0)
        if float(row["Df_cm"]) == 0 or float(row["I_A"]) == 0:
            continue

        Df_cm = float(row["Df_cm"])
        Ue_V = float(row["Ue_V"])
        I_A = float(row["I_A"])

        # Compute radius
        Re = radius_from_deflection(Df_cm)

        # Magnetic field
        B_T = a_T_per_A * I_A

        # e/m
        em_val = compute_em(Ue_V, B_T, Re)

        results.append(EMResult(row, Re, B_T, em_val))

    return results


def results_to_dataframe(results: list[EMResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Df_cm": r.df_row["Df_cm"],
            "Ue_V": r.df_row["Ue_V"],
            "I_A": r.df_row["I_A"],
            "Re_m": r.Re_m,
            "B_T": r.B_T,
            "em_SI": r.em_value,
        })
    return pd.DataFrame(rows)


# === Main ======================================================================

def main():
    # --- Load CSVs ---
    df_small = load_csv("thomson_petites_bobines.csv")
    df_large = load_csv("thomson_grandes_bobines.csv")

    # --- Process datasets ---
    results_small = process_dataset(df_small, A_SMALL_T_per_A)
    results_large = process_dataset(df_large, A_LARGE_T_per_A)

    df_small_res = results_to_dataframe(results_small)
    df_large_res = results_to_dataframe(results_large)

    # --- Compute statistics (Type A before Student correction) ---
    mean_small = df_small_res["em_SI"].mean()
    mean_large = df_large_res["em_SI"].mean()

    std_small = df_small_res["em_SI"].std(ddof=1)
    std_large = df_large_res["em_SI"].std(ddof=1)

    print("=== Petites bobines ===")
    print(df_small_res)
    print(f"moyenne e/m = {mean_small:.3e}")
    print(f"écart-type   = {std_small:.3e}")
    print()

    print("=== Grandes bobines ===")
    print(df_large_res)
    print(f"moyenne e/m = {mean_large:.3e}")
    print(f"écart-type   = {std_large:.3e}")
    print()


if __name__ == "__main__":
    main()

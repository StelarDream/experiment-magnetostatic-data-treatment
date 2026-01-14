# scr/constants.py
import math
import numpy as np

# --- Instrument uncertainties ---

# Teslameter jitter → directly observed fluctuation amplitude
TESLAMETER_U_B_MT: float = 0.03  # mT (instrumental noise)

# Visual/manual readout → uniform distribution → RES / sqrt(3)
POSITION_RULE_RES_MM: float = 1.0  # mm (ruler's graduation)
POSITION_U_Z_MM: float = POSITION_RULE_RES_MM / math.sqrt(3.0)  # ≈ 0.577 mm

# Digital display with stable last digit → resolution-limited → RES / 2
CURRENT_RES_A: float = 0.01  # A (last displayed digit)
CURRENT_U_I_A: float = CURRENT_RES_A / 2.0  # = 0.005 A

# --- Physical constants ---

MU0_SI = 4 * np.pi * 1e-7

# --- Coil geometry ---

# Pair 1 and 2: grandes bobines (µ0 + Thomson grande config, same for both)
N_GRANDES_BOBINES: int = 157
R_GRANDES_BOBINES_M: float = 210.0 / 1000.0
U_R_GRANDES_RES_MM = 5.0  # ±5 mm (visual on marked support)
U_R_GRANDES_M = (U_R_GRANDES_RES_MM / 1000.0) / math.sqrt(3)

# Pair 3: petite bobines (Thomson petite config)
# N_PETITE_BOBINES: int = ...  # not taken, nor is it needed
R_PETITE_BOBINES_M: float = 61.0 / 1000.0
U_R_PETITES_RES_MM = 0.5  # ±0.5 mm (caliper measurement)
U_R_PETITES_M = (U_R_PETITES_RES_MM / 1000.0) / math.sqrt(3)
# scr/constants.py
import math
import numpy as np

# --- Instrument uncertainties ---
TESLAMETER_U_B_MTESLA: float = 0.03  # absolute uncertainty on B, in mT

POSITION_RULE_RES_MM: float = 1.0
POSITION_U_Z_MM: float = POSITION_RULE_RES_MM / math.sqrt(3.0)  # type B, uniform law

MU0_SI = 4 * np.pi * 1e-7

N_GRANDES_BOBINES: int = 157    # number of turns per coil
R_GRANDES_BOBINES_MM: float = 210.0    # radius in mm
R_GRANDES_BOBINES_M: float = R_GRANDES_BOBINES_MM / 1000.0

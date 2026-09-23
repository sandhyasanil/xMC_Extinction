"""xMC_Extinction: analytic R(V)-dependent extinction curves for SMC, LMC and
LMC2 dust, in the Cardelli, Clayton & Mathis (1989) functional form."""
from _common import BaseMCExtinction, ccm_optical_nir, OPT_BREAK, A_BREAK, B_BREAK
from LMC import LMCExtinction
from LMC2 import LMC2Extinction
from SMC import SMCExtinction

__all__ = ["SMCExtinction", "LMCExtinction", "LMC2Extinction",
           "BaseMCExtinction", "ccm_optical_nir", "OPT_BREAK", "A_BREAK", "B_BREAK"]
__version__ = "1.0.0"

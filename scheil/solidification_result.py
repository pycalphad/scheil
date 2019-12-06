import numpy as np


class SolidificationResult():
    """Data from an equilibrium or Scheil-Gulliver solidification simulation.

    Parameters
    ----------
    x_liquid : Dict[str, List[float]]
        Mapping of component name to composition at each temperature.
    fraction_solid : List[float]
        Fraction of solid at each temperature.
    temperatures : List[float]
        List of simulation temperatures.
    phase_amounts : Dict[str, float]
        Map of {phase_name: amount_list} for solid phases where amount_list is
        a list of instantaneus phase amounts at each temperature. Should be
        less than 1 unless the solidification all occured in 1 step (e.g.
        solidification at the eutectic composition)
    converged : bool
        For Scheil: True if the liquid stopping criteria was met. False otherwise
        For equilibrium: True if no liquid remains, False otherwise.

    Attributes
    ----------
    x_liquid : Dict[str, List[float]
    fraction_solid : List[float]
    temperatures : List[float]
    phase_amounts : Dict[str, float]
    fraction_liquid : List[float]
        Fraction of liquid at each temperature (convenience for 1-fraction_solid)
    cum_phase_amounts : Dict[str, list]
        Map of {phase_name: amount_list} for solid phases where amount_list is
        a list of cumulative phase amounts at each temperature.

    """

    def __init__(self, x_liquid, fraction_solid, temperatures, phase_amounts, converged):
        self.x_liquid = x_liquid
        self.fraction_solid = fraction_solid
        self.fraction_liquid = (1.0 - np.array(fraction_solid)).tolist()
        self.temperatures = temperatures
        self.phase_amounts = phase_amounts
        self.cum_phase_amounts = {ph: np.cumsum(amnts).tolist() for ph, amnts in phase_amounts.items()}
        self.converged = converged

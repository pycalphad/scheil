import numpy as np

class SolidifcationResult():
    """Short summary.

    Parameters
    ----------
    temperatures : list
        List of simulation temperatures.
    x_liquid : list
        Mole fraction of the liquid phase at each temperature.
    fraction_solid : list
        Fraction of liquid at each temperature.
    phase_amounts : dict
        Dictionary of {phase_name: amount_list} where amount_list is a list of
        instantaneus phase amounts at each temperature. Should be less than 1
        unless the solidification all occured in 1 step (e.g. solidification
        at the eutectic composition)
    converged : bool
        For Scheil: True if the liquid stopping criteria was met. False otherwise
        For equilibrium: True if no liquid remains, False otherwise.

    Attributes
    ----------
    fraction_liquid : list
        Description of attribute `fraction_liquid`.
    x_liquid
    fraction_solid
    temperatures
    phase_amounts
    cum_phase_amounts : Dict[str, list]
        Cumulative phase amounts )

    """
    def __init__(self, x_liquid, fraction_solid, temperatures, phase_amounts, converged):
        self.x_liquid = x_liquid
        self.fraction_solid = fraction_solid
        self.fraction_liquid = (1.0-np.array(fraction_solid)).tolist()
        self.temperatures = temperatures
        self.phase_amounts = phase_amounts
        self.cum_phase_amounts = {ph: np.cumsum(amnts).tolist() for ph, amnts in phase_amounts.items()}
        self.converged = converged

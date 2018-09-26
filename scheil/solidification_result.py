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
        cumulative phase amounts at each temperature.

    Attributes
    ----------
    fraction_liquid : list
        Description of attribute `fraction_liquid`.
    x_liquid
    fraction_solid
    temperatures
    phase_amounts

    """
    def __init__(self, x_liquid, fraction_solid, temperatures, phase_amounts):
        self.x_liquid = x_liquid
        self.fraction_solid = fraction_solid
        self.fraction_liquid = (1.0-np.array(fraction_solid)).tolist()
        self.temperatures = temperatures
        self.phase_amounts = phase_amounts

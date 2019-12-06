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
    method : str
        Method used to create the solidification result, should be "scheil" or "equilibrium"

    Attributes
    ----------
    x_liquid : Dict[str, List[float]
    fraction_solid : List[float]
    temperatures : List[float]
    phase_amounts : Dict[str, float]
    method : str
    fraction_liquid : List[float]
        Fraction of liquid at each temperature (convenience for 1-fraction_solid)
    cum_phase_amounts : Dict[str, list]
        Map of {phase_name: amount_list} for solid phases where amount_list is
        a list of cumulative phase amounts at each temperature.

    """

    def __init__(self, x_liquid, fraction_solid, temperatures, phase_amounts, converged, method):
        self.x_liquid = x_liquid
        self.fraction_solid = fraction_solid
        self.fraction_liquid = (1.0 - np.array(fraction_solid)).tolist()
        self.temperatures = temperatures
        self.phase_amounts = phase_amounts
        self.cum_phase_amounts = {ph: np.cumsum(amnts).tolist() for ph, amnts in phase_amounts.items()}
        self.converged = converged
        self.method = method

    def __repr__(self):
        name = self.__class__.__name__
        temps = f"T=({max(self.temperatures):0.1f} -> {min(self.temperatures):0.1f})"
        phases_with_nonzero_amount = "(" + ", ".join(sorted([ph for ph, amnt in self.cum_phase_amounts.items() if amnt[-1] > 0])) + ")"
        return f"<{name}: {self.method} {temps} {phases_with_nonzero_amount}>"

    def to_dict(self):
        d = {
            'x_liquid': self.x_liquid,
            'fraction_solid': self.fraction_solid,
            'temperatures': self.temperatures,
            'phase_amounts': self.phase_amounts,
            'converged': self.converged,
            'method': self.method,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        x_liquid = d['x_liquid']
        fraction_solid = d['fraction_solid']
        temperatures = d['temperatures']
        phase_amounts = d['phase_amounts']
        converged = d['converged']
        method = d['method']
        return cls(x_liquid, fraction_solid, temperatures, phase_amounts, converged, method)

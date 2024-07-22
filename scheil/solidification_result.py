import numpy as np
import pandas as pd


class SolidificationResult():
    """Data from an equilibrium or Scheil-Gulliver solidification simulation.

    Parameters
    ----------
    phase_compositions : Mapping[PhaseName, Mapping[ComponentName, List[float]]]
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
    phase_compositions : Mapping[PhaseName, Mapping[ComponentName, List[float]]]
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

    def __init__(self, phase_compositions, fraction_solid, temperatures, phase_amounts, converged, method):
        # sort of a hack because we don't explictly track liquid phase name
        self.phase_compositions = phase_compositions
        self.fraction_solid = fraction_solid
        self.fraction_liquid = (1.0 - np.array(fraction_solid)).tolist()
        self.temperatures = temperatures
        self.phase_amounts = phase_amounts
        self.cum_phase_amounts = {ph: np.cumsum(amnts).tolist() for ph, amnts in phase_amounts.items()}
        self.liquid_phase_name = list(set(self.phase_compositions.keys()) - set(self.cum_phase_amounts.keys()))[0]
        self.x_liquid = phase_compositions[self.liquid_phase_name]  # keeping for backwards compatibility, but this is also present in self.phase_compositions
        self.converged = converged
        self.method = method

    def __repr__(self):
        name = self.__class__.__name__
        temps = f"T=({max(self.temperatures):0.1f} -> {min(self.temperatures):0.1f})"
        phases_with_nonzero_amount = "(" + ", ".join(sorted([ph for ph, amnt in self.cum_phase_amounts.items() if amnt[-1] > 0])) + ")"
        return f"<{name}: {self.method} {temps} {phases_with_nonzero_amount}>"

    def to_dict(self):
        d = {
            'phase_compositions': self.phase_compositions,
            'fraction_solid': self.fraction_solid,
            'temperatures': self.temperatures,
            'phase_amounts': self.phase_amounts,
            'converged': self.converged,
            'method': self.method,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        phase_compositions = d['phase_compositions']
        fraction_solid = d['fraction_solid']
        temperatures = d['temperatures']
        phase_amounts = d['phase_amounts']
        converged = d['converged']
        method = d['method']
        return cls(phase_compositions, fraction_solid, temperatures, phase_amounts, converged, method)

    def to_dataframe(self, include_zero_phases=True):
        """
        Parameters
        ----------
        include_zero_phases : Optional[bool]
            If True (the default), phases that never become stable in the simulation will be included.
        """
        data_dict = {}
        data_dict["Temperature (K)"] = self.temperatures
        data_dict[f"NP({self.liquid_phase_name})"] = self.fraction_liquid
        stable_phases = {self.liquid_phase_name}
        for phase_name, vals in sorted(self.cum_phase_amounts.items()):
            if vals[-1] > 0: # vals[-2] handles liquid case
                stable_phases.add(phase_name)
            if phase_name in stable_phases or include_zero_phases:
                data_dict[f"NP({phase_name})"] = vals
        for phase_name, phase_compositions in self.phase_compositions.items():
            if phase_name in stable_phases or include_zero_phases:
                for comp, vals in phase_compositions.items():
                    data_dict[f"X({phase_name},{comp})"] = vals
        df = pd.DataFrame(data_dict)
        return df

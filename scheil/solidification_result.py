import numpy as np

class SolidifcationResult():
    def __init__(self, x_liquid, fraction_solid, temperatures, phase_amounts):
        self.x_liquid = x_liquid
        self.fraction_solid = fraction_solid
        self.fraction_liquid = (1.0-np.array(fraction_solid)).tolist()
        self.temperatures = temperatures
        self.phase_amounts = phase_amounts

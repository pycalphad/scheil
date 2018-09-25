
from pycalphad import equilibrium, variables as v
from scheil.build_callables import build_callables
import numpy as np
import collections

NAN_DICT = collections.defaultdict(lambda:np.nan)  # Dictionary that always returns NaN for any key

def simulate_scheil_solidification(dbf, comps, phases, composition, start_temperature, liquid_phase_name='LIQUID'):
    callables = build_callables(dbf, comps, phases)
    solid_phases = sorted(set(phases)-{'GAS', liquid_phase_name})
    temp = start_temperature
    independent_comps = sorted(composition.keys())

    x_liquid = [composition]
    fraction_solid = [0.0]
    temperatures = [temp]

    while fraction_solid[-1] < 1:
        conds = {v.T: temp, v.P: 101325}
        conds.update(x_liquid[-1])
        eq = equilibrium(dbf, comps, phases, conds, **callables)
        if 'LIQUID' not in eq.Phase.isel(T=0,P=0).values:
            break
        # TODO: Will break if there is a liquid miscibility gap
        liquid_vertex = sorted(np.nonzero(eq.Phase.isel(T=0,P=0).values.flat == 'LIQUID'))[0]
        liquid_comp = {comp: float(eq.X.isel(T=0,P=0,vertex=liquid_vertex).sel(component=str(comp)[2:]).values) for comp in independent_comps}
        x_liquid.append(liquid_comp)
        current_fraction_solid = float(fraction_solid[-1])
        for solid_phase in solid_phases:
            if solid_phase not in eq.Phase.isel(T=0,P=0).values:
                continue
            # TODO: Will break if there is a miscibility gap
            solid_vertex = sorted(np.nonzero(eq.Phase.isel(T=0,P=0).values.flat == solid_phase))[0]
            solid_comp = {comp: float(eq.X.isel(T=0,P=0,vertex=solid_vertex).sel(component=str(comp)[2:]).values) for comp in independent_comps}
            delta_comp = liquid_comp[independent_comps[0]] - solid_comp[independent_comps[0]]
            delta_liquid_comp = x_liquid[-1][independent_comps[0]] - x_liquid[-2][independent_comps[0]]
            delta_fraction_solid = (1-current_fraction_solid) * delta_liquid_comp / delta_comp
            current_fraction_solid += delta_fraction_solid

        fraction_solid.append(current_fraction_solid)
        temperatures.append(temp)
        temp -= 1
    if fraction_solid[-1] < 1:
        x_liquid.append(NAN_DICT)
        fraction_solid.append(1.0)
        temperatures.append(temp)
    return x_liquid, fraction_solid, temperatures


def simulate_equilibrium_solidification(dbf, comps, phases, composition, start_temperature, end_temperature, step_temperature, liquid_phase_name='LIQUID', callables=None):
    # Compute the equilibrium solidification path
    callables = build_callables(dbf, comps, phases)
    conds = {v.T: np.arange(start_temperature, end_temperature, step_temperature), v.P: 101325}
    conds.update(composition)
    eq = equilibrium(dbf, comps, phases, conds, **callables)
    return eq

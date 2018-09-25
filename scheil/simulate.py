
from pycalphad import Database, equilibrium, variables as v
from scheil.build_callables import build_callables
import numpy as np
import collections

dbf = Database('alzn_mey.tdb')
comps = ['AL', 'ZN', 'VA']
phases = sorted(dbf.phases.keys())

liquid_phase_name = 'LIQUID'
initial_composition = {v.X('ZN'): 0.3}
start_temperature = 850 # K, Needs to be at or above the liquidus temperature

callables = build_callables(dbf, comps, phases)
solid_phases = sorted(set(phases)-{'GAS', liquid_phase_name})
temp = start_temperature
initial_comp = initial_composition

x_liquid = [initial_comp]
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
    liquid_comp = {comp: float(eq.X.isel(T=0,P=0,vertex=liquid_vertex).sel(component=str(comp)[2:]).values) for comp in initial_comp.keys()}
    x_liquid.append(liquid_comp)
    current_fraction_solid = float(fraction_solid[-1])
    for solid_phase in solid_phases:
        if solid_phase not in eq.Phase.isel(T=0,P=0).values:
            continue
        # TODO: Will break if there is a miscibility gap
        solid_vertex = sorted(np.nonzero(eq.Phase.isel(T=0,P=0).values.flat == solid_phase))[0]
        solid_comp = {comp: float(eq.X.isel(T=0,P=0,vertex=solid_vertex).sel(component=str(comp)[2:]).values) for comp in initial_comp.keys()}
        delta_comp = liquid_comp[sorted(initial_comp.keys())[0]] - solid_comp[sorted(initial_comp.keys())[0]]
        delta_liquid_comp = x_liquid[-1][sorted(initial_comp.keys())[0]] - x_liquid[-2][sorted(initial_comp.keys())[0]]
        delta_fraction_solid = (1-current_fraction_solid) * delta_liquid_comp / delta_comp
        current_fraction_solid += delta_fraction_solid

    fraction_solid.append(current_fraction_solid)
    temperatures.append(temp)
    temp -= 1
if fraction_solid[-1] < 1:
    x_liquid.append(collections.defaultdict(lambda:np.nan))
    fraction_solid.append(1.0)
    temperatures.append(temp)

# Compute the equilibrium solidification path
conds = {v.T: temperatures, v.P: 101325}
conds.update(initial_comp)
eq = equilibrium(dbf, comps, phases, conds, **callables)

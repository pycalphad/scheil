
from pycalphad import equilibrium, variables as v
from scheil.build_callables import build_callables
from scheil import SolidifcationResult
import numpy as np
import collections

NAN_DICT = collections.defaultdict(lambda:np.nan)  # Dictionary that always returns NaN for any key

def simulate_scheil_solidification(dbf, comps, phases, composition,
                                   start_temperature, step_temperature=1.0,
                                   liquid_phase_name='LIQUID'):
    """Perform a Scheil-Gulliver solidification simulation.

    Parameters
    ----------
    dbf : pycalphad.Database
        Database object.
    comps : list
        List of components in the system.
    phases : list
        List of phases in the system.
    composition : dict
        Dictionary of independent `v.X` composition variables.
    start_temperature : float
        Starting temperature for simulation. Must be single phase liquid.
    step_temperature : float, optional
        Temperature step size. Defaults to 1.0.
    liquid_phase_name : str
        Name of the phase treated as liquid.

    Returns
    -------
    scheil.solidification_result.SolidifcationResult

    """
    callables = build_callables(dbf, comps, phases)
    solid_phases = sorted(set(phases)-{'GAS', liquid_phase_name})
    temp = start_temperature
    independent_comps = sorted(composition.keys(), key=str)

    x_liquid = [composition]
    fraction_solid = [0.0]
    temperatures = [temp]
    phase_amounts = {ph: [0.0] for ph in solid_phases}

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
                phase_amounts[solid_phase].append(0.0)
                continue
            # TODO: Will break if there is a miscibility gap
            solid_vertex = sorted(np.nonzero(eq.Phase.isel(T=0,P=0).values.flat == solid_phase))[0]
            solid_comp = {comp: float(eq.X.isel(T=0,P=0,vertex=solid_vertex).sel(component=str(comp)[2:]).values) for comp in independent_comps}
            delta_comp = liquid_comp[independent_comps[0]] - solid_comp[independent_comps[0]]
            delta_liquid_comp = x_liquid[-1][independent_comps[0]] - x_liquid[-2][independent_comps[0]]
            delta_fraction_solid = (1-current_fraction_solid) * delta_liquid_comp / delta_comp
            current_fraction_solid += delta_fraction_solid
            phase_amounts[solid_phase].append(delta_fraction_solid)

        fraction_solid.append(current_fraction_solid)
        temperatures.append(temp)
        temp -= step_temperature
    if fraction_solid[-1] < 1:
        x_liquid.append(NAN_DICT)
        fraction_solid.append(1.0)
        temperatures.append(temp)
        # set the final phase amount to the phase fractions in the eutectic
        # this method gives the sum total phase amounts of 1.0 by construction
        for solid_phase in solid_phases:
            amount = np.nansum(eq.NP.isel(T=0,P=0).where(eq.Phase == solid_phase).values)
            phase_amounts[solid_phase].append(float(amount)*(1-current_fraction_solid))

    # take the instantaneous phase amounts and make them cumulative
    phase_amounts = {ph: np.cumsum(amnts).tolist() for ph, amnts in phase_amounts.items()}

    return SolidifcationResult(x_liquid, fraction_solid, temperatures, phase_amounts)


def simulate_equilibrium_solidification(dbf, comps, phases, composition,
                                        start_temperature, end_temperature, step_temperature,
                                        liquid_phase_name='LIQUID', callables=None):
    # Compute the equilibrium solidification path
    solid_phases = sorted(set(phases)-{'GAS', liquid_phase_name})
    independent_comps = sorted(composition.keys(), key=str)
    callables = build_callables(dbf, comps, phases)
    conds = {v.T: (end_temperature, start_temperature, step_temperature), v.P: 101325}
    conds.update(composition)
    eq = equilibrium(dbf, comps, phases, conds, **callables)

    temperatures = eq["T"].values.tolist()
    x_liquid = []
    fraction_solid = []
    phase_amounts = {ph: [] for ph in solid_phases}
    for T_idx in reversed(range(len(temperatures))):
        curr_eq = eq.isel(T=T_idx, P=0)
        curr_fraction_solid = 0.0
        # calculate the phase amounts
        for solid_phase in solid_phases:
            amount = float(np.nansum(curr_eq.NP.where(curr_eq.Phase == solid_phase).values))
            phase_amounts[solid_phase].append(amount)
            curr_fraction_solid += amount
        fraction_solid.append(curr_fraction_solid)

        # liquid phase constitution
        if 'LIQUID' in curr_eq.Phase.values:
            # TODO: will break for liquid miscibility gap
            liquid_vertex = sorted(np.nonzero(curr_eq.Phase.values.flat == 'LIQUID'))[0]
            liquid_comp = {comp: float(curr_eq.X.isel(vertex=liquid_vertex).sel(component=str(comp)[2:]).values) for comp in independent_comps}
            x_liquid.append(liquid_comp)
        else:
            x_liquid.append(np.nan)

    return SolidifcationResult(x_liquid, fraction_solid, temperatures, phase_amounts)

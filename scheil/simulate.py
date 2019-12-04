import numpy as np
from collections import defaultdict
from pycalphad import equilibrium, variables as v
from pycalphad.codegen.callables import build_callables
from pycalphad.core.utils import instantiate_models, generate_dof, \
    unpack_components
from .solidification_result import SolidifcationResult
from .utils import order_disorder_dict, local_sample, order_disorder_eq_phases

NAN_DICT = defaultdict(lambda: np.nan)


def simulate_scheil_solidification(dbf, comps, phases, composition,
                                   start_temperature, step_temperature=1.0,
                                   liquid_phase_name='LIQUID', eq_kwargs=None,
                                   stop=0.0001, verbose=False, adaptive=True):
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
    step_temperature : Optional[float]
        Temperature step size. Defaults to 1.0.
    liquid_phase_name : Optional[str]
        Name of the phase treated as liquid.
    eq_kwargs: Optional[Dict[str, Any]]
        Keyword arguments for equilibrium
    stop: Optional[float]
        Stop when the phase fraction of liquid is below this amount.
    adaptive: Optional[bool]
        Whether

    Returns
    -------
    SolidifcationResult

    """
    eq_kwargs = eq_kwargs or dict()
    STEP_SCALE_FACTOR = 1.2  # How much to try to adapt the temperature step by
    MAXIMUM_STEP_SIZE_REDUCTION = 5.0
    T_STEP_ORIG = step_temperature
    models = instantiate_models(dbf, comps, phases)
    if verbose:
        print('building callables... ', end='')
    cbs = build_callables(dbf, comps, phases, models, additional_statevars={v.P, v.T, v.N}, build_gradients=True, build_hessians=True)
    if verbose:
        print('done')
    solid_phases = sorted(set(phases) - {'GAS', liquid_phase_name})
    temp = start_temperature
    independent_comps = sorted(composition.keys(), key=str)
    x_liquid = [composition]
    fraction_solid = [0.0]
    temperatures = [temp]
    phase_amounts = {ph: [0.0] for ph in solid_phases}
    ord_disord_dict = order_disorder_dict(dbf, comps, phases)

    if adaptive and ('points' in eq_kwargs.get('calc_opts', {})):
        # Dynamically add points as the simulation runs
        species = unpack_components(dbf, comps)
        dof_dict = {ph: generate_dof(dbf.phases[ph], species)[1] for ph in phases}
    else:
        adaptive = False

    converged = False
    phases_seen = {liquid_phase_name, ''}
    while fraction_solid[-1] < 1:
        conds = {v.T: temp, v.P: 101325.0, v.N: 1.0}
        comp_conds = x_liquid[-1]
        fmt_comp_conds = ', '.join(['{}={:0.2f}'.format(c, val) for c, val in comp_conds.items()])
        conds.update(comp_conds)
        eq = equilibrium(dbf, comps, phases, conds, callables=cbs, model=models, **eq_kwargs)
        if adaptive:
            points_dict = eq_kwargs['calc_opts']['points']
            for vtx in range(eq.vertex.size):
                masked = eq.isel(vertex=vtx)
                ph = str(masked.Phase.values.squeeze())
                pts = points_dict.get(ph)
                if pts is not None:
                    if verbose:
                        print(f'Adding points to {ph}.', end=' ')
                    dof = dof_dict[ph]
                    points_dict[ph] = np.concatenate([pts, local_sample(masked.Y.values.squeeze()[:sum(dof)].reshape(1, -1), dof, pdens=20)], axis=0)

        eq_phases = order_disorder_eq_phases(eq, ord_disord_dict)
        num_eq_phases = np.nansum(eq_phases != '')
        new_phases_seen = set(eq_phases).difference(phases_seen)
        if len(new_phases_seen) > 0:
            if verbose:
                print('New phases seen: {}.'.format(new_phases_seen), end=' ')
            phases_seen |= new_phases_seen
            # temp += step_temperature
            # step_temperature = T_STEP_ORIG
            # continue
        if liquid_phase_name not in eq["Phase"].values.squeeze():
            if num_eq_phases == 0:
                print('Convergence failure: T={} and {}'.format(temp, fmt_comp_conds), end=' ')
            if T_STEP_ORIG / step_temperature > MAXIMUM_STEP_SIZE_REDUCTION:
                if verbose:
                    print('No liquid phase found at T={}, {} (Found {}). Maximum step size reduction exceeded. Stopping.'.format(temp, fmt_comp_conds, eq_phases))
                converged = False
                break
            else:
                if verbose:
                    print('No liquid phase found at T={}, {} (Found {}). Stepping back and reducing step size.'.format(temp, fmt_comp_conds, eq_phases))
                temp += step_temperature
                step_temperature /= STEP_SCALE_FACTOR
                continue
        # TODO: Will break if there is a liquid miscibility gap
        liquid_vertex = sorted(np.nonzero(eq["Phase"].values.squeeze().flat == liquid_phase_name))[0]
        liquid_comp = {comp: float(eq["X"].isel(vertex=liquid_vertex).squeeze().sel(component=str(comp)[2:]).values) for comp in independent_comps}
        x_liquid.append(liquid_comp)
        np_liq = np.nansum(eq.where(eq["Phase"] == liquid_phase_name).NP.values)
        current_fraction_solid = float(fraction_solid[-1])
        found_phase_amounts = [(liquid_phase_name, np_liq)]  # tuples of phase name, amount
        for solid_phase in solid_phases:
            if solid_phase not in eq_phases:
                phase_amounts[solid_phase].append(0.0)
                continue
            np_tieline = np.nansum(eq.isel(vertex=eq_phases.index(solid_phase))["NP"].values.squeeze())
            found_phase_amounts.append((solid_phase, np_tieline))
            delta_fraction_solid = (1 - current_fraction_solid) * np_tieline
            current_fraction_solid += delta_fraction_solid
            phase_amounts[solid_phase].append(delta_fraction_solid)
        fraction_solid.append(current_fraction_solid)
        temperatures.append(temp)
        NL = 1 - fraction_solid[-1]
        if verbose:
            phase_amnts = ' '.join(['NP({})={:0.3f}'.format(ph, amnt) for ph, amnt in found_phase_amounts])
            if NL < 1.0e-3:
                print('T={:0.3f}, {}, ΔT={:0.3f}, NL: {:.2E}, {}'.format(temp, fmt_comp_conds, step_temperature, NL, phase_amnts), end=' ')
            else:
                print('T={:0.3f}, {}, ΔT={:0.3f}, NL: {:0.3f}, {}'.format(temp, fmt_comp_conds, step_temperature, NL, phase_amnts), end=' ')
        if NL < stop:
            if verbose:
                print('Liquid fraction below criterion {} . Stopping at fmt_comp_conds'.format(stop))
            converged = True
            break
        if verbose:
            print()  # add line break
        temp -= step_temperature

    if fraction_solid[-1] < 1:
        x_liquid.append(NAN_DICT)
        fraction_solid.append(1.0)
        temperatures.append(temp)
        # set the final phase amount to the phase fractions in the eutectic
        # this method gives the sum total phase amounts of 1.0 by construction
        for solid_phase in solid_phases:
            if solid_phase in eq_phases:
                amount = np.nansum(eq.isel(vertex=eq_phases.index(solid_phase))["NP"].values.squeeze())
                phase_amounts[solid_phase].append(float(amount) * (1 - current_fraction_solid))
            else:
                phase_amounts[solid_phase].append(0.0)

    return SolidifcationResult(x_liquid, fraction_solid, temperatures, phase_amounts, converged)


def simulate_equilibrium_solidification(dbf, comps, phases, composition,
                                        start_temperature, end_temperature, step_temperature,
                                        liquid_phase_name='LIQUID', verbose=False):
    # Compute the equilibrium solidification path
    solid_phases = sorted(set(phases) - {'GAS', liquid_phase_name})
    independent_comps = sorted(composition.keys(), key=str)
    models = instantiate_models(dbf, comps, phases)
    if verbose:
        print('building callables... ', end='')
    cbs = build_callables(dbf, comps, phases, models, additional_statevars={v.P, v.T, v.N}, build_gradients=True, build_hessians=True)
    if verbose:
        print('done')
    conds = {v.T: (end_temperature, start_temperature, step_temperature), v.P: 101325}
    conds.update(composition)
    eq = equilibrium(dbf, comps, phases, conds, callables=cbs)

    temperatures = eq["T"].values.tolist()
    x_liquid = []
    fraction_solid = []
    phase_amounts = {ph: [] for ph in solid_phases}  # instantaneous phase amounts
    cum_phase_amounts = {ph: [] for ph in solid_phases}
    for T_idx in reversed(range(len(temperatures))):
        curr_eq = eq.isel(T=T_idx, P=0)
        curr_fraction_solid = 0.0
        # calculate the phase amounts
        for solid_phase in solid_phases:
            amount = float(np.nansum(curr_eq.NP.where(curr_eq.Phase == solid_phase).values))
            # Since the equilibrium calculations always give the "cumulative" phase amount,
            # we need to take the difference to get the instantaneous.
            cum_phase_amounts[solid_phase].append(amount)
            if len(phase_amounts[solid_phase]) == 0:
                phase_amounts[solid_phase].append(amount)
            else:
                phase_amounts[solid_phase].append(amount - cum_phase_amounts[solid_phase][-2])
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

    converged = np.isclose(fraction_solid[-1], 1.0)
    return SolidifcationResult(x_liquid, fraction_solid, temperatures, phase_amounts, converged)

import numpy as np
from sympy import Symbol
from pycalphad import Model, variables as v
from pycalphad.core.sympydiff_utils import build_functions
from pycalphad.core.utils import unpack_kwarg, unpack_components
from pycalphad.core.phase_rec import PhaseRecord_from_cython

def get_pure_elements(dbf, comps):
    """
    Return a list of pure elements in the system

    Parameters
    ----------
    dbf : pycalphad.Database
        A Database object
    comps : list
        A list of component names (species and pure elements)

    Returns
    -------
    list
        A list of pure elements in the Database
    """
    comps = sorted(unpack_components(dbf, comps))
    components = [x for x in comps]
    desired_active_pure_elements = [list(x.constituents.keys()) for x in components]
    desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
    pure_elements = sorted(set([x for x in desired_active_pure_elements if x != 'VA']))
    return pure_elements

# TODO: move this into pycalphad, see https://github.com/pycalphad/pycalphad/issues/189
def build_callables(dbf, comps, phases, model=None, param_symbols=None, output='GM', build_gradients=True):
    """
    Create a dictionary of callable dictionaries for phases in equilibrium

    Parameters
    ----------
    dbf : pycalphad.Database
        A pycalphad Database object
    comps : list
        List of component names
    phases : list
        List of phase names
    model : dict or type
        Dictionary of {phase_name: Model subclass} or a type corresponding to a
        Model subclass. Defaults to ``Model``.
    param_symbols : list
        SymPy Symbol objects that will be preserved in the callable functions.
    output : str
        Output property of the particular Model to sample
    build_gradients : bool
        Whether or not to build gradient functions. Defaults to True.

    Returns
    -------
    dict
        Dictionary of keyword argument callables to pass to equilibrium.

    Notes
    -----
    Based on the pycalphad equilibrium method for building phases as of commit 37ff75ce.

    Examples
    --------
    >>> dbf = Database('AL-NI.tdb')
    >>> comps = ['AL', 'NI', 'VA']
    >>> phases = ['FCC_L12', 'BCC_B2', 'LIQUID', 'AL3NI5', 'AL3NI2', 'AL3NI']
    >>> eq_callables = eq_callables_dict(dbf, comps, phases)
    >>> equilibrium(dbf, comps, phases, conditions, **eq_callables)
    """
    comps = sorted(unpack_components(dbf, comps))
    pure_elements = get_pure_elements(dbf, comps)

    eq_callables = {
        'massfuncs': {},
        'massgradfuncs': {},
        'callables': {},
        'grad_callables': {},
        'hess_callables': {},
    }

    models = unpack_kwarg(model, default_arg=Model)
    param_symbols = param_symbols if param_symbols is not None else []
    # wrap param symbols in Symbols if they are strings
    if all([isinstance(sym, string_types) for sym in param_symbols]):
        param_symbols = [Symbol(sym) for sym in param_symbols]
    param_values = np.zeros_like(param_symbols, dtype=np.float64)

    phase_records = {}
    # create models
    # starting from pycalphad
    for name in phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name)
        site_fracs = mod.site_fractions
        variables = sorted(site_fracs, key=str)
        try:
            out = getattr(mod, output)
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))

        # Build the callables of the output
        # Only force undefineds to zero if we're not overriding them
        undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable) - set(param_symbols))
        for undef in undefs:
            out = out.xreplace({undef: float(0)})
        build_output = build_functions(out, tuple([v.P, v.T] + site_fracs), parameters=param_symbols, include_grad=build_gradients)
        if build_gradients:
            cf, gf = build_output
        else:
            cf = build_output
            gf = None
        hf = None
        eq_callables['callables'][name] = cf
        eq_callables['grad_callables'][name] = gf
        eq_callables['hess_callables'][name] = hf

        # Build the callables for mass
        # TODO: In principle, we should also check for undefs in mod.moles()

        if build_gradients:
            mcf, mgf = zip(*[build_functions(mod.moles(el), [v.P, v.T] + variables,
                                           include_obj=True,
                                           include_grad=build_gradients,
                                           parameters=param_symbols)
                           for el in pure_elements])
        else:
            mcf = tuple([build_functions(mod.moles(el), [v.P, v.T] + variables,
                                           include_obj=True,
                                           include_grad=build_gradients,
                                           parameters=param_symbols)
                           for el in pure_elements])
            mgf = None
        eq_callables['massfuncs'][name] = mcf
        eq_callables['massgradfuncs'][name] = mgf

        # creating the phase records triggers the compile
        phase_records[name.upper()] = PhaseRecord_from_cython(comps, variables,
                                                           np.array(dbf.phases[name].sublattices, dtype=np.float),
                                                           param_values, eq_callables['callables'][name],
                                                           eq_callables['grad_callables'][name], eq_callables['hess_callables'][name],
                                                           eq_callables['massfuncs'][name], eq_callables['massgradfuncs'][name])

    # finally, add the models to the eq_callables
    eq_callables['model'] = dict(models)
    return eq_callables


"""
Utilities for distinguishing and renaming ordered and disordered configurations of
multi-sublattice phases.

`OrderingRecord` objects are able to be used for any phase. `OrderingRecords` can be
created automatically for phases modeled with a partitioned order/disorder model through
the `create_ordering_records` method, since the partitioned model contains all the
information about the ordered and disordered phase.
"""

from dataclasses import dataclass
from typing import Sequence, List
import itertools
from collections import defaultdict
import numpy as np
import xarray as xr
from pycalphad.core.utils import unpack_species

@dataclass
class OrderingRecord:
    ordered_phase_name: str
    disordered_phase_name: str
    subl_dof: Sequence[int]  # number of degrees of freedom in each sublattice of the ordered phase
    symmetric_subl_idx: Sequence[Sequence[int]]  # List of sublattices (of the ordered phase) that are symmetric

    def is_disordered(self, site_fractions):
        # Short circuit if any site fraction is NaN (i.e. no phase or a different phase)
        if np.any(np.isnan(site_fractions[:sum(self.subl_dof)])):
            return False

        # For each sublattice, create a `slice` object for slicing the site
        # fractions of that particular sublattice from the site fraction array
        subl_slices = []
        for subl_idx in range(len(self.subl_dof)):
            start_idx = np.sum(self.subl_dof[:subl_idx], dtype=np.int_)
            end_idx = start_idx + self.subl_dof[subl_idx]
            subl_slices.append(slice(start_idx, end_idx))

        # For each set of symmetrically equivalent sublattices
        for symm_subl in self.symmetric_subl_idx:
            # Check whether the site fractions of each pair of symmetrically
            # equivalent sublattices are ordered or disordered
            for idx1, idx2 in itertools.combinations(symm_subl, 2):
                # A phase is ordered if any pair of sublattices does not have
                # equal (within numerical tolerance) site fractions
                pair_is_ordered = np.any(~np.isclose(site_fractions[subl_slices[idx1]], site_fractions[subl_slices[idx2]]))
                if pair_is_ordered:
                    return False
        return True


def create_ordering_records(dbf, comps, phases):
    """Return a dictionary with the sublattice degrees of freedom and equivalent
    sublattices for order/disorder phases

    Parameters
    ----------
    dbf : pycalphad.Database
    comps : list[str]
        List of active components to consider
    phases : list[str]
        List of active phases to consider

    Returns
    -------
    List[OrderingRecord]

    Notes
    -----
    Phases which should be checked for ordered/disordered configurations are
    determined heuristically for this script.

    The heuristic for a phase satisfies the following:
    1. The phase is the ordered part of an order-disorder model
    2. The equivalent sublattices have all the same number of elements
    """
    species = unpack_species(dbf, comps)
    ordering_records = []
    for phase_name in phases:
        phase_obj = dbf.phases[phase_name]
        if phase_name == phase_obj.model_hints.get('ordered_phase', ''):
            # This phase is active and modeled with an order/disorder model.
            dof = [len(subl.intersection(species)) for subl in phase_obj.constituents]
            # Define the symmetrically equivalent sublattices as any sublattices
            # TODO: the heuristic here is simple and incorrect for cases like L1_2.
            # that have the same site ratio. Create a {site_ratio: [subl idx]} dict
            site_ratio_idxs = defaultdict(lambda: [])
            for subl_idx, site_ratio in enumerate(phase_obj.sublattices):
                site_ratio_idxs[site_ratio].append(subl_idx)
            equiv_sublattices = list(site_ratio_idxs.values())
            ordering_records.append(OrderingRecord(phase_name, phase_obj.model_hints['disordered_phase'], dof, equiv_sublattices))
    return ordering_records


def rename_disordered_phases(eq_result, ordering_records):
    """
    Modify an xarray Dataset to rename the ordered phase names to the disordered phase
    names if the equilibrium configuration is disordered

    Parameters
    ----------
    eq_result : xarray.Dataset
    order_disorder_dict : OrderingRecord
        Output from scheil.utils.order_disorder_dict

    Returns
    -------
    xrray.Dataset
        Dataset modified in-place

    Notes
    -----
    This function does _not_ change the site fractions array of the disordered
    configurations to match the site fractions matching the internal degrees of freedom
    of the disordered phase's constituents (although that should be possible).

    Examples
    --------
    >>> from pycalphad import Database, equilibrium, variables as v
    >>> import pycalphad.tests.databases
    >>> from importlib.resources import files
    >>> dbf = Database(str(files(pycalphad.tests.databases).joinpath("alcfe_b2.tdb")))
    >>> comps = ['AL', 'FE', 'VA']
    >>> phases = list(dbf.phases.keys())
    >>> eq_res = equilibrium(dbf, comps, ['B2_BCC'], {v.P: 101325, v.T: 1000, v.N: 1, v.X('AL'): [0.1, 0.4]})
    >>> ordering_records = create_ordering_records(dbf, comps, phases)
    >>> eq_res.Phase.values.squeeze().tolist()
    [['B2_BCC', '', ''], ['B2_BCC', '', '']]
    >>> out_result = rename_disordered_phases(eq_res, ordering_records)
    >>> eq_res.Phase.values.squeeze().tolist()
    [['A2_BCC', '', ''], ['B2_BCC', '', '']]
    """

    for ord_rec in ordering_records:
        # Array indices matching phase with ordered phase name
        mask = eq_result.Phase == ord_rec.ordered_phase_name
        # disordered_mask is a boolean mask that is True if the element listed as an
        # ordered phase is a disordered configuration. We want to broadcast over all
        # dimensions except for internal_dof (we need all internal dof to determine if
        # the site fractions are disordered). The `OrderingRecord.is_disordered` method
        # is not vectorized (operates on 1D site fractions), so we use `vectorize=True`.
        disordered_mask = xr.apply_ufunc(ord_rec.is_disordered, eq_result.where(mask).Y, input_core_dims=[['internal_dof']], vectorize=True)
        # Finally, use `xr.where` to set the value of the phase name to the disordered
        # phase everywhere the mask is true and use the existing value otherwise
        eq_result['Phase'] = xr.where(disordered_mask, ord_rec.disordered_phase_name, eq_result.Phase)
    return eq_result
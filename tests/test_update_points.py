import numpy as np
from pycalphad import Workspace, variables as v
from scheil.simulate import _update_points
from fixtures import select_database, load_database


def _setup(dbf):
    """Return a point-equilibrium Workspace and matching empty points/dof dicts."""
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['LIQUID', 'FCC_A1', 'HCP_A3'],
                    {v.T: 700, v.P: 101325, v.N: 1, v.X('ZN'): 0.3})
    dof_dict = {ph: list(map(len, mod.constituents)) for ph, mod in wks.models.items()}
    points_dict = {ph: np.empty((0, sum(dof))) for ph, dof in dof_dict.items()}
    return wks, points_dict, dof_dict


@select_database("alzn_mey.tdb")
def test_update_points_appends_equilibrium_site_fractions(load_database):
    """The point appended by _update_points must be the equilibrium site fractions, not the state variables."""
    wks, points_dict, dof_dict = _setup(load_database())
    # Single-phase FCC_A1 region for this condition
    assert [cs.phase_record.phase_name for cs in wks.get_composition_sets()] == ['FCC_A1']

    _update_points(wks, points_dict, dof_dict, local_pdens=0)

    assert points_dict['FCC_A1'].shape == (1, 2)
    expected = [wks.get(v.Y('FCC_A1', 0, 'AL')), wks.get(v.Y('FCC_A1', 0, 'ZN'))]
    np.testing.assert_allclose(points_dict['FCC_A1'][0], expected, atol=1e-8)
    # Other phases were not stable, so nothing should be appended
    assert points_dict['LIQUID'].shape == (0, 2)
    assert points_dict['HCP_A3'].shape == (0, 2)


@select_database("alzn_mey.tdb")
def test_update_points_local_samples_are_valid_site_fractions(load_database):
    """Locally sampled points must be valid site fractions and include the equilibrium point itself."""
    wks, points_dict, dof_dict = _setup(load_database())

    _update_points(wks, points_dict, dof_dict, local_pdens=20)

    pts = points_dict['FCC_A1']
    assert pts.shape == (21, 2)  # 20 local samples + the equilibrium point
    assert np.all(pts >= 0) and np.all(pts <= 1)
    np.testing.assert_allclose(pts.sum(axis=1), 1.0)
    expected = [wks.get(v.Y('FCC_A1', 0, 'AL')), wks.get(v.Y('FCC_A1', 0, 'ZN'))]
    assert np.any(np.all(np.isclose(pts, expected, atol=1e-8), axis=1))

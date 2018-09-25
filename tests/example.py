from pycalphad import Database, variables as v
from scheil.simulate import simulate_scheil_solidification

dbf = Database('alzn_mey.tdb')
comps = ['AL', 'ZN', 'VA']
phases = sorted(dbf.phases.keys())

liquid_phase_name = 'LIQUID'
initial_composition = {v.X('ZN'): 0.3}
start_temperature = 850 # K, Needs to be at or above the liquidus temperature

sol_res = simulate_scheil_solidification(dbf, comps, phases, initial_composition, start_temperature)
print(sol_res[0])


import matplotlib.pyplot as plt
plt.plot(temperatures, fraction_solid, label='Scheil')
plt.plot(temperatures, 1-np.nansum(eq.where(eq.Phase=='LIQUID')['NP'].values, axis=-1).flatten(), label='EQ')
plt.legend(loc='best')
plt.xlabel('Temperature (K)')
plt.ylabel('Fraction of Solid')
plt.show()

plt.plot(temperatures, [x[v.X('ZN')] for x in x_liquid], label='Scheil')
eq_compositions = np.nansum(eq.where(eq.Phase=='LIQUID').X.sel(component='ZN'), axis=-1).flatten()
# Hack, since these should be completely solid
eq_compositions[np.where(eq_compositions==0)] = np.nan
plt.plot(temperatures, eq_compositions, label='EQ')
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Zn Composition of Liquid')
plt.show()

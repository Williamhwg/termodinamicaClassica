#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
table18 = pd.read_csv('table18.txt', index_col = 'i', sep=' ', dtype={'xi': 'float', 'yi': 'float'})
table19 = pd.read_csv('table19.txt', index_col = 'i', sep=' ', dtype={'Ni': 'float', 'ti': 'float', 'di': 'float', 'li': 'float', 'hi': 'float','bi': 'float', 'gi': 'float', 'ei': 'float'})
table19.columns = ['Ni','ti','di','li','hi','bi','gi','ei']
# %%
# Constants
Tc = 514.71 # K
rho_c = 5.93 # mol/dm³
R = 8.31446262 #J/(mol.K)
# %%
T = np.linspace(100,700,1000)
rho = np.linspace(1, 22, 1000)
tau = Tc/T
delta = rho/rho_c
# %%
xdelta, xtau = np.meshgrid(delta,tau)
# %%
def alpha_0(delta, tau): # J/mol
    sum_alpha0 = np.cumsum([table18['xi'][i]*(1-np.log(1-np.exp(-table18['yi'][i]*tau))) for i in range(4,8)], axis=1)[-1]
    other_terms0 = np.log(delta) + table18['xi'][1]*np.log(tau) + table18['xi'][2] + table18['xi'][3]*tau
    return (other_terms0 + sum_alpha0)*T*R

def alpha_r(delta, tau):
    sum_r1 = np.cumsum([table19['Ni'][i]*delta**(table19['di'][i])*tau**(table19['ti'][i]) for i in range(1,7)], axis=1)[-1]
    sum_r2 = np.cumsum([table19['Ni'][i]*delta**(table19['di'][i])*tau**(table19['ti'][i])*np.exp(-delta**(table19['li'][i])) for i in range(7,17)], axis=1)[-1]
    sum_r3 = np.cumsum([table19['Ni'][i]*delta**(table19['di'][i])*tau**(table19['ti'][i])*np.exp((-table19['hi'][i]*(delta-table19['ei'][i])**2-table19['bi'][i]*(tau-table19['gi'][i])**2)) for i in range(17,26)], axis=1)[-1]
    return sum_r1 + sum_r2 + sum_r3

def helm(delta, tau):
    return alpha_0(delta, tau) + alpha_r(delta,tau)
# %%
plt.contourf(xdelta*rho_c, Tc/xtau, helm(xdelta,xtau), levels=50, cmap='RdBu')
plt.xlabel('Density (mol/dm³)')
plt.ylabel('Temperature (K)')
plt.colorbar()
# %%
plt.contourf(xdelta*rho_c, Tc/xtau, np.gradient(helm(xdelta,xtau))[0], levels=50, cmap='RdBu')
plt.colorbar()
# %%
np.gradient(np.gradient(helm(xdelta,xtau))[-1])[-1]
# %%
plt.contour(xdelta*rho_c, Tc/xtau,np.gradient(np.gradient(helm(xdelta,xtau))[-1])[-1] , levels=100, cmap='RdBu')
plt.xlabel('Density (mol/dm³)')
plt.ylabel('Temperature (K)')
plt.colorbar()

# %%
hessians = np.asarray(np.gradient(np.gradient(helm(xdelta,xtau))))
hessians.shape
hessians[1:].shape
# plt.contour(xdelta*rho_c, Tc/xtau, hessians[0][0] , levels=100, cmap='RdBu')
# plt.xlabel('Density (mol/dm³)')
# plt.ylabel('Temperature (K)')
# plt.colorbar()
# %%
for d in range(50):
    plt.plot(T, helm(d, tau))
# %%

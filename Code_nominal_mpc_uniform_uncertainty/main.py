# %%
from template_simulator import template_simulator
from template_mpc import template_mpc
from template_model import template_model
import matplotlib.gridspec as gridspec
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import time
#from demand_700 import demand_700
from demand_700_test import demand_700
# %%

""" User settings: """
show_animation = False
store_results = False
n_sim_steps = 24*7
"""
Set global parameters :

"""


"""
Get solar_power and demand data: :

"""
solar_power_unc, solar_power_baseline, demand_baseline, demand_unc, _, _ = demand_700()

# %%
"""
Get configured do-mpc modules:

"""
model, B_cap, p_ano, p_water, I, A_el, Volt_disch, Volt_charg, N_cell = template_model()
mpc = template_mpc(model, B_cap, p_ano, p_water, I, A_el, Volt_disch, Volt_charg, N_cell, solar_power_baseline, demand_baseline)
simulator = template_simulator(model, solar_power_unc, demand_baseline)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""
P_hyd_tank_0 = 101325
E_battery_0 = B_cap*3600*0.9
x0 = np.array([P_hyd_tank_0, E_battery_0])

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

z0=np.array([0.0734,0.000176,0.000208,0.105,68.07,75194,0.0011622,0.00233,0.0368,0.0531,75406.8,23.217,12022])
mpc.z0 = z0
simulator.z0 = z0
estimator.z0 = z0


mpc.set_initial_guess()
simulator.set_initial_guess()
# %%
"""
Setup graphic:
"""
# %%
#fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data,[],['C_out_oxy_ano'])
graphics = do_mpc.graphics.Graphics(mpc.data)
# Create figure with arbitrary Matplotlib method
fig, ax = plt.subplots(2, sharex=True, figsize=(8, 9))
# Configure plot (pass the previously obtained ax objects):
graphics.add_line(var_type='_aux', var_name='Power_disca', axis=ax[0])
graphics.add_line(var_type='_aux', var_name='mol_produced', axis=ax[1])
graphics.add_line(var_type='_tvp', var_name='n_dot_demand', axis=ax[1])

# graphics.add_line(var_type='_aux', var_name='Power_charg_aux', axis=ax[2])
# graphics.add_line(var_type='_aux', var_name='Power_elect_aux', axis=ax[3])
# graphics.add_line(var_type='_aux', var_name='Power_disch_aux', axis=ax[4])
# graphics.add_line(var_type='_aux', var_name='H2_impurity', axis=ax[5])
# graphics.add_line(var_type='_x', var_name='P_hyd_tank', axis=ax[6])
# graphics.add_line(var_type='_tvp', var_name='n_dot_demand', axis=ax[7])
# graphics.add_line(var_type='_aux', var_name='mol_produced', axis=ax[7])
# graphics.add_line(var_type='_aux', var_name='Power_disca', axis=ax[8])
# graphics.add_line(var_type='_aux', var_name='Volt', axis=ax[9])

ax[0].set_ylabel('P_disc [kW]')
ax[1].set_ylabel('Production/ Demand [mol/s]')


ax[1].set_xlabel('time [sec]')

# ax[5].set_ylabel('n_dot_demand')
# ax[6].set_ylabel('P_tank')
# ax[7].set_ylabel('Production/ Demand')
# ax[8].set_ylabel('Power_disca')
# ax[9].set_ylabel('Volt')
# ax[9].set_xlabel('time [h]')
for axes in ax:
    axes.ticklabel_format(style='plain', useOffset=False)
plt.ion()
fig.align_ylabels()
fig.tight_layout()
#plt.xticks(np.arange(0, 3600*7*24+1, 3600*5), np.arange(0, 7*24/5+1, 1))

# %%
"""
Run MPC main loop:
"""

start_time = time.time()

for k in range (n_sim_steps):
    u0 = mpc.make_step(x0)
    #print(u0)
    #Power_solar_simulator = simulator.data['_tvp', 'Power_solar']
    Power_solar_simulator = solar_power_unc[k]
    Power_solar_mpc = solar_power_baseline[k]
    #print(Power_solar_simulator)
    if Power_solar_mpc != 0:
        u0[0] = (u0[0]/Power_solar_mpc)*Power_solar_simulator
        u0[1] = (u0[1]/Power_solar_mpc)*Power_solar_simulator
        
    if u0[1] > 0.1 and u0[2]>0.1:
        u0[1] = 0

    y_next = simulator.make_step(u0)
    #print(y_next)
    x0 = estimator.make_step(y_next)
    #print(x0)
    print('current time step= '+str(k))

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

end_time = time.time()
duration = end_time - start_time
print("Script execution time:"+ str(duration)+ "seconds")

input('Press any key to exit.')



# %%
"""
Graphs and plots:
"""
# configure to get results for control-loop
#fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5, 10))
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5, 3.3))
#fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6.5, 2.1))


# ax[0].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'SOC'], linestyle='dashed',color='blue')
# ax[0].plot(simulator.data['_time']/(60*60), simulator.data['_aux', 'SOC'], color='green')
# ax[0].axhline(y=90, color='r', linestyle='dotted')
# ax[0].axhline(y=20, color='r', linestyle='dotted')
# ax[0].set_ylabel('$SOC [\%]$')

# ax[-1].set_xlabel('Time [h]')

# ax[1].plot(mpc.data['_time']/(60*60), mpc.data['_x', 'P_hyd_tank']/101325, linestyle='dashed',color='blue')
# ax[1].plot(simulator.data['_time']/(60*60), simulator.data['_x', 'P_hyd_tank']/101325, color='green')
# ax[1].set_ylabel('$p$ $[bar]$')
# ax[1].axhline(y=400, color='r', linestyle='dotted')
# ax[1].axhline(y=1, color='r', linestyle='dotted')

# ax[2].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_elect']/1000, linestyle='dashed',color='blue')
# ax[2].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_elect']/1000, color='green')
# ax[2].set_ylabel('$P_{elect} [kW]$')

# # ax[2].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Total_Power_elect']/1000)
# # ax[2].plot(simulator.data['_time']/(60*60),simulator.data['_aux', 'Total_Power_elect']/1000)
# # ax[2].set_ylabel('$P_{tot,elect} [kW]$')

# ax[3].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_charg']/1000, linestyle = 'dashed',color='blue')
# ax[3].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_charg']/1000, color='green', label='$P_{charg}$')
# #ax[3].set_ylabel('$P_{charg} [kW]$')

# ax[3].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_disch']/1000, linestyle='dashed',color='black')
# ax[3].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_disch']/1000,label='$P_{disch}$', color='grey')
# ax[3].set_ylabel('$P_{charg}/P_{disch} [kW]$')
# ax[3].legend()

# ax[4].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Power_solar']/1000, linestyle='dashed',color='blue')
# ax[4].plot(simulator.data['_time']/(60*60),simulator.data['_aux', 'Power_solar']/1000, color='green', label = '$P_{solar}$')
# ax[4].set_ylabel('$P_{solar} [kW]$')


# ax[4].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Power_disca'], linestyle='dashed',color='black')
# ax[4].plot(simulator.data['_time']/(60*60),simulator.data['_aux', 'Power_disca'], color='grey', label = '$P_{discard}$')
# ax[4].set_ylabel('$P_{solar}/P_{discard} [kW]$')
# ax[4].legend()

# ax[5].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Current_density'],color='blue', linestyle='dashed' )
# ax[5].plot(simulator.data['_time']/(60*60), simulator.data['_aux', 'Current_density'], color='green')
# ax[5].set_ylabel('$J [A \cdot m^{-2}]$')
# ax[5].axhline(y=5000, color='r', linestyle='dotted')
###################################



ax[0].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'H2_impurity'], linestyle='dashed',color='blue')
ax[0].plot(simulator.data['_time']/(60*60),
           simulator.data['_aux', 'H2_impurity'], color='green')
ax[0].axhline(y=2, color='r', linestyle='dotted')
ax[0].set_ylabel('$H_2$ Impurity $[vol\%]$')

ax[1].plot(mpc.data['_time']/(60*60), mpc.data['_z', 'C_out_hyd_cath'], linestyle='dashed',color='blue')
ax[1].plot(simulator.data['_time']/(60*60), simulator.data['_z', 'C_out_hyd_cath'], color='green')
ax[1].set_ylabel('$C_{out,H_2}^{cat} [mol \cdot m^{-3}]$')
ax[-1].set_xlabel('Time [h]')
#####################################



for axes in ax:
   axes.ticklabel_format(style='plain', useOffset=False)

plt.tight_layout()
fig.align_ylabels()
plt.savefig('test_result.pdf')
plt.show()




# %%

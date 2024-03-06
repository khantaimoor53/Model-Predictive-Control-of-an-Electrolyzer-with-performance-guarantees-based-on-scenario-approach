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
show_animation = True
store_results = False
n_sim_steps = 24*7
"""
Set global parameters :

"""


"""
Get solar_power and demand data: :

"""
solar_power_unc, solar_power_baseline, demand_baseline, demand_unc, _, _ = demand_700()


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

#mpc.u0 = np.array([0,0,30000*20])

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
fig, ax = plt.subplots(6, sharex=True, figsize=(7, 9))

# Configure plot (pass the previously obtained ax objects):
graphics.add_line(var_type='_u', var_name='Power_elect', axis=ax[0])
graphics.add_line(var_type='_aux', var_name='Total_Power_elect', axis=ax[1])
graphics.add_line(var_type='_aux', var_name='H2_impurity', axis=ax[2])
graphics.add_line(var_type='_aux', var_name='SOC', axis=ax[3])
graphics.add_line(var_type='_aux', var_name='Power_solar', axis=ax[4])
graphics.add_line(var_type='_tvp', var_name='n_dot_demand', axis=ax[5])

ax[0].set_ylabel('P_el')
ax[1].set_ylabel('TP')
ax[2].set_ylabel('H2_impurity')
ax[3].set_ylabel('SOC')
ax[4].set_ylabel('Solar_p')
ax[5].set_ylabel('n_dot_demand')

for axes in ax:
    axes.ticklabel_format(style='plain', useOffset=False)
plt.ion()
fig.align_ylabels()
fig.tight_layout()
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

    if show_animation == True:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

end_time = time.time()
duration = end_time - start_time
print("Script execution time:"+ str(duration)+ "seconds")

input('Press any key to exit.')


# Store results:
if store_results == True:
    do_mpc.data.save_results([mpc, simulator], 'Electrolyzer_results')


# %%
"""
Graphs and plots:
"""

fig, ax = plt.subplots(10, 1, sharex=True, figsize=(7, 9))

ax[0].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'SOC'])
ax[0].plot(simulator.data['_time']/(60*60), simulator.data['_aux', 'SOC'])
ax[0].set_ylabel('SOC')
ax[-1].set_xlabel('Time [h]')

# ax[0].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+13*3600,3600)/3600,mpc.data.prediction(('_x','E_battery')).reshape(-1,1))
# ax[1].plot(mpc.data['_time']/(60*60),mpc.data['_aux','Power_dc_p'])
# ax[1].set_ylabel('P_solar')

#ax[1].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Total_Power']/20)
#ax[1].plot((simulator.data['_time']/(60*60)),simulator.data['_aux', 'Total_Power']/20)
#ax[1].set_ylabel('Tot_P [W]')

#ax[1].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_aux','Total_Power')).reshape(-1,1))

ax[2].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_charg'])
ax[2].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_charg'])
ax[2].set_ylabel('P_ch')

#ax[2].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_u','Power_charg_p')).reshape(-1,1))


ax[3].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_elect'])
ax[3].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_elect'])
ax[3].set_ylabel('P_el')

#ax[3].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_u','Power_elect_p')).reshape(-1,1))


ax[4].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'H2_impurity'])
ax[4].plot(simulator.data['_time']/(60*60),
           simulator.data['_aux', 'H2_impurity'])
ax[4].set_ylabel('Imp [vol%]]')

#ax[4].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_aux','H2_impurity')).reshape(-1,1))

ax[5].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'Power_solar'])
ax[5].plot(simulator.data['_time']/(60*60),
           simulator.data['_tvp', 'Power_solar'])
ax[5].set_ylabel('P_solar')


#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'n_dot_demand'])
#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'n_dot_out'])
#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_x', 'P_hyd_tank']/101325)
ax[6].plot(simulator.data['_time']/(60*60), simulator.data['_x', 'P_hyd_tank']/101325)
#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_disch'])
#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Current_density'])
#plt.legend(['n_dot_demand','n_dot_out'])
ax[7].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'n_dot_demand'])
ax[8].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Power_disca'])
ax[9].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'mol_produced']*20)
ax[9].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'n_dot_demand'])
ax[9].plot((simulator.data['_time']/(60*60)), simulator.data['_tvp', 'n_dot_demand'])

for axes in ax:
    axes.ticklabel_format(style='plain', useOffset=False)


fig.align_axes()
fig.tight_layout()

plt.show()




# %%

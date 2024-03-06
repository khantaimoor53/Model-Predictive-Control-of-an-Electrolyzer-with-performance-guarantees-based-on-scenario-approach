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
fig, ax = plt.subplots(5, sharex=True, figsize=(8, 9))
# Configure plot (pass the previously obtained ax objects):
graphics.add_line(var_type='_aux', var_name='Power_solar_scenario', axis=ax[0])
graphics.add_line(var_type='_aux', var_name='SOC', axis=ax[1])
#graphics.add_line(var_type='_tvp', var_name='n_dot_demand', axis=ax[1])

graphics.add_line(var_type='_aux', var_name='Current_density', axis=ax[2])
# graphics.add_line(var_type='_aux', var_name='Power_elect_aux', axis=ax[3])
# graphics.add_line(var_type='_aux', var_name='Power_disch_aux', axis=ax[4])
# graphics.add_line(var_type='_aux', var_name='H2_impurity', axis=ax[5])
graphics.add_line(var_type='_x', var_name='P_hyd_tank', axis=ax[2])
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

    if show_animation==True:
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
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'Electrolyzer_results')


# %%
"""
Graphs and plots:
"""
#fig, ax = plt.subplots(6, 1, sharex=True, figsize=(6.5, 10))
#fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5, 3.3))
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6.5, 2.1))


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



# ax[0].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'H2_impurity'], linestyle='dashed',color='blue')
# ax[0].plot(simulator.data['_time']/(60*60),
#            simulator.data['_aux', 'H2_impurity'], color='green')
# ax[0].axhline(y=2, color='r', linestyle='dotted')
# ax[0].set_ylabel('$H_2$ Impurity $[vol\%]$')

# ax[1].plot(mpc.data['_time']/(60*60), mpc.data['_z', 'C_out_hyd_cath'], linestyle='dashed',color='blue')
# ax[1].plot(simulator.data['_time']/(60*60), simulator.data['_z', 'C_out_hyd_cath'], color='green')
# ax[1].set_ylabel('$C_{out,H_2}^{cat} [mol \cdot m^{-3}]$')
# ax[-1].set_xlabel('Time [h]')
#####################################

ax.plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'n_dot_demand'], linestyle='dashed',color='blue')
ax.plot(simulator.data['_time']/(60*60), simulator.data['_tvp', 'n_dot_demand'], color='green', label = '$\dot{n}_{demand}$')
#ax.set_ylabel('$\dot{n}_{demand} [mol/s]$')
ax.set_xlabel('Time [h]')
ax.plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'mol_produced'], linestyle='dashed',color='black')
ax.plot(simulator.data['_time']/(60*60), simulator.data['_aux', 'mol_produced'], color='grey', label = '$\dot{n}_{out,H_2}$')
ax.set_ylabel('$\dot{n}_{out,H_2/demand} [mol\cdot s^{-1}]$')
ax.legend()
# ax[1].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Total_Power_elect']/20)
# ax[1].plot((simulator.data['_time']/(60*60)),simulator.data['_aux', 'Total_Power_elect']/20)
# ax[1].set_ylabel('Tot_P [W]')

# ax[1].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Total_Power_elect']/20)
# ax[1].plot((simulator.data['_time']/(60*60)),simulator.data['_aux', 'Total_Power_elect']/20)
# ax[1].set_ylabel('Tot_P [W]')
#ax[1].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_aux','Total_Power_elect')).reshape(-1,1))

# ax[2].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_charg']/1000)
# ax[2].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_charg']/1000)
# ax[2].set_ylabel('P_ch [kW]')
# ax[4].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Power_solar']/1000)
# ax[4].plot(simulator.data['_time']/(60*60),simulator.data['_aux', 'Power_solar']/1000)
# ax[4].set_ylabel('P_solar [kW]')

#ax[2].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_u','Power_charg_p')).reshape(-1,1))


# ax[3].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_elect']/1000)
# ax[3].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_elect']/1000)
# ax[3].set_ylabel('P_el [kW]')
#ax[3].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'H2_impurity'])
#ax[3].plot(simulator.data['_time']/(60*60),
           #simulator.data['_aux', 'H2_impurity'])
# ax[3].axhline(y=2, color='r', linestyle='dashed')
#ax[3].set_ylabel('Imp [vol%]')

# ax[4].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_disch']/1000)
# ax[4].plot(simulator.data['_time']/(60*60),simulator.data['_u', 'Power_disch']/1000)
# ax[4].set_ylabel('P_disch [kW]')

# ax[5].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'Power_solar'], label = 'mpc')
# ax[5].plot(simulator.data['_time']/(60*60),
#            simulator.data['_tvp', 'Power_solar'], label = 'sim')
# ax[5].set_ylabel('P_solar')
#ax[3].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_u','Power_elect_p')).reshape(-1,1))


# ax[4].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'H2_impurity'])
# ax[4].plot(simulator.data['_time']/(60*60),
#            simulator.data['_aux', 'H2_impurity'])
# #ax[4].axhline(y=2, color='r', linestyle='-')
# ax[4].set_ylabel('Imp [vol%]')

#ax[4].plot(np.arange(mpc.data['_time'][-1],mpc.data['_time'][-1]+12*3600,3600)/3600,mpc.data.prediction(('_aux','H2_impurity')).reshape(-1,1))




#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'n_dot_demand'])
#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'n_dot_out'])
#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_x', 'P_hyd_tank']/101325)

#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_u', 'Power_disch'])
#ax[6].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Current_density'])
#plt.legend(['n_dot_demand','n_dot_out'])
# ax[7].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'n_dot_demand'])
# ax[8].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'Power_disca'])
# ax[9].plot(mpc.data['_time']/(60*60), mpc.data['_aux', 'mol_produced']*20)
# ax[9].plot(mpc.data['_time']/(60*60), mpc.data['_tvp', 'n_dot_demand'])
# ax[9].plot((simulator.data['_time']/(60*60))+1, simulator.data['_tvp', 'n_dot_demand'])



# for axes in ax:
#     axes.ticklabel_format(style='plain', useOffset=False)

plt.tight_layout()
fig.align_ylabels()
plt.savefig('test_result_ms.pdf')
plt.show()




# %%


# epsilon= []
# f1 = []
# f2 = []
# f3 = []
# f4 = []
# for i in np.arange(0,1,0.0001):
#     epsilon.append(i)
#     f1.append((40*39*i*(1-i)**38))
#     f2.append((20*19*i*(1-i)**18))
#     f3.append((10*9*i*(1-i)**8))
#     f4.append((5*4*i*(1-i)**3))

# plt.plot(epsilon,f1, label='$N_{sample} = 40$')
# plt.plot(epsilon,f2, label='$N_{sample} = 20$')
# plt.plot(epsilon,f3, label='$N_{sample} = 10$')
# plt.plot(epsilon,f4, label='$N_{sample} = 5$')

# plt.ylabel('$f_{SA}(\epsilon)$')
# plt.xlabel('$\epsilon$')
# plt.legend()
# plt.savefig('effect_of_N_sample_on_fSA.pdf')



# epsilon= []
# f1 = []
# f2 = []
# f3 = []
# f4 = []
# for i in np.arange(0,1,0.0001):
#     epsilon.append(i)
   
#     f1.append((30*29*i*(1-i)**28))
#     f2.append(0.5*30*29*28*i*i*(1-i)**27)
#     f3.append((1/6)*30*29*28*27*i*i*i*(1-i)**26)
#     f4.append((5/(5*4*3*2))*30*29*28*27*26*i*i*i*i*(1-i)**25)

# plt.plot(epsilon,f1, label='$d = 2$')
# plt.plot(epsilon,f2, label='$d = 3$')
# plt.plot(epsilon,f3, label='$d = 4$')
# plt.plot(epsilon,f4, label='$d = 5$')

# plt.ylabel('$f_{SA}(\epsilon)$')
# plt.xlabel('$\epsilon$')
# plt.legend()
# plt.savefig('effect_of_d_on_fSA.pdf')
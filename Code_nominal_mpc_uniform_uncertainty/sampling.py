# %%
from do_mpc import *
import numpy as np
import pandas as pd
from demand_700 import demand_700
from template_simulator import template_simulator
from template_mpc import template_mpc
from template_model import template_model
import multiprocessing as mp
import time
start_time = time.time()

######## Settings/ Configuration ########

n_sim_steps = 24*7
n_processes = 1

######### Sampling #########

plan = tools.load_pickle('./sampling/plan3.pkl')

print('We will sample '+ str(len(plan))+ ' samples')
time.sleep(5)

# %%
def closed_loop(solar_power_baseline,solar_power_unc, demand_baseline, demand_unc):

    model, B_cap, p_ano, p_water, I, A_el, Volt_disch, Volt_charg, N_cell = template_model()
    mpc = template_mpc(model, B_cap, p_ano, p_water, I, A_el, Volt_disch, Volt_charg, N_cell,solar_power_baseline, demand_baseline)
    simulator = template_simulator(model,solar_power_unc, demand_baseline) # not considering demand uncertainty by providing it a baseline value
    

    mpc.reset_history()
    simulator.reset_history()
    #estimator.reset_history()

    P_hyd_tank_0 = 101325
    E_battery_0 = B_cap*3600*0.9
    x0 = np.array([P_hyd_tank_0, E_battery_0])

    mpc.x0 = x0
    simulator.x0 = x0
    #estimator.x0 = x0

    z0=np.array([0.0734,0.000176,0.000208,0.105,68.07,75194,0.0011622,0.00233,0.0368,0.0531,75406.8,23.217,12022])
    mpc.z0 = z0
    simulator.z0 = z0
    #estimator.z0 = z0

    #mpc.u0 = np.array([0,0,30000])

    mpc.set_initial_guess()
    simulator.set_initial_guess()

    for k in range (n_sim_steps):

        u0 = mpc.make_step(x0)

        Power_solar_simulator = solar_power_unc[k]
        Power_solar_mpc = solar_power_baseline[k]
        #print(Power_solar_simulator)
        if Power_solar_mpc != 0:
            u0[0] = (u0[0]/Power_solar_mpc)*Power_solar_simulator
            u0[1] = (u0[1]/Power_solar_mpc)*Power_solar_simulator
        
        if u0[1] > 0.1 and u0[2]>0.1:
            u0[1] = 0
        

        x0 = simulator.make_step(u0)
        #x0 = estimator.make_step(y_next)
        print('current time step= '+str(k))
    
    return simulator.data, mpc.data


def main():
    sampler = sampling.Sampler(plan)
    sampler.set_param(overwrite = True)
    sampler.set_sample_function(closed_loop)
    sampler.data_dir = './results/'
    start_time = time.time()

    with mp.Pool(processes=n_processes) as Pool:
        p = Pool.map(sampler.sample_idx, list(range(sampler.n_samples)))
        
    end_time = time.time()
    duration = end_time - start_time
    print("Script execution time:"+ str(duration)+ "seconds")



if __name__ == "__main__":
    main()
# %%
#####################

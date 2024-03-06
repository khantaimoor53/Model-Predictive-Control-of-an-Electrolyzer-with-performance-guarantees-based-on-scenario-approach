# %%
from do_mpc import *
import numpy as np
import pandas as pd
from demand_700 import demand_700
from template_simulator import template_simulator
from template_mpc import template_mpc
from template_model import template_model
from matplotlib import pyplot as plt
import time
from distribution_m_shaped_high1 import inverse_CDF_m_shaped

start_time = time.time()

######## Settings/ Configuration ########

n_samples = 200


######### Generate sampling plan #########
# %%
sp = sampling.SamplingPlanner()
sp.set_param(overwrite = False)
sp.data_dir = './sampling/'

sp.set_sampling_var('solar_power_baseline')
sp.set_sampling_var('solar_power_unc')
sp.set_sampling_var('demand_baseline')
sp.set_sampling_var('demand_unc')

_,_,demand_baseline_val,_,irradiation_val, panel_area = demand_700()



for i in range(n_samples):

    solar_unc_val = np.zeros(len(irradiation_val))
    for j in range(len(solar_unc_val)):
        solar_unc_val[j] = inverse_CDF_m_shaped(np.random.uniform(0,1))
    solar_unc_val[j] = round(solar_unc_val[j],3)
    plan = sp.add_sampling_case(solar_power_baseline=irradiation_val* panel_area*0.22,
                                solar_power_unc=  solar_unc_val* irradiation_val * panel_area*0.22,
                                demand_baseline=demand_baseline_val,
                                demand_unc=np.random.uniform(0.6 * demand_baseline_val, 1.4 * demand_baseline_val))

sp.export("plan")
# %%

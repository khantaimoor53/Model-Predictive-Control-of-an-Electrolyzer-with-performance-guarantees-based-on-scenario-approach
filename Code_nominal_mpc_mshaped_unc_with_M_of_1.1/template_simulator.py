# %%
import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *



def template_simulator(model,solar_power_unc, demand_unc):

    """
    Simulator:
    """
    ########## Settings/ Configuration ##########
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'idas',
        't_step': 1*60*60
    }

    simulator.set_param(**params_simulator)

    ########## Uncertinity/ tvp values ##########

    tvp_num = simulator.get_tvp_template()
    

    def tvp_fun(t_now):
        
        tvp_num['Power_solar'] = solar_power_unc[int(((t_now)/3600))]
        tvp_num['n_dot_demand'] = demand_unc[int(((t_now)/3600))]
        
        return tvp_num

    simulator.set_tvp_fun(tvp_fun)
    simulator.setup()
    
    return simulator

# %%

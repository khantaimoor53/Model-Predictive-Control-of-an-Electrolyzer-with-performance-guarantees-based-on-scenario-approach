# %%
import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc
      

def template_mpc(model, B_cap, p_ano, p_water, I, A_el, Volt_disch, Volt_charg, N_cell,solar_power_baseline, demand_baseline):
    
    """
    Controller:
    """
    ########## Settings/ Configuration ##########

    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 24,
        
        'n_robust': 0,
        'open_loop': 0,
        't_step': 1*60*60,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        'store_full_solution': False,
        'nlpsol_opts': {'ipopt.print_level':0}, #'ipopt.linear_solver': 'MA97'}
        #'nlpsol_opts':{'ipopt.linear_solver':'MA27','ipopt.max_iter':2000},
        #'nlpsol_opts': {'ipopt.linear_solver':'MA97','ipopt.max_iter': 2000 ,'ipopt.acceptable_dual_inf_tol':1e-9, 'ipopt.print_level': 5, 'print_time': 0, 'ipopt.acceptable_tol': 1e-5, 'ipopt.acceptable_obj_change_tol': 1e-2}
        
    }

    mpc.set_param(**setup_mpc)

    ########## constraints ##########

    # State constraints:

    mpc.bounds['lower', '_x', 'E_battery'] = 0.2*B_cap*3600
    mpc.bounds['lower', '_x', 'P_hyd_tank'] = 101325

    mpc.bounds['lower', '_z', 'P_out_oxy_ano'] = 0.0
    mpc.bounds['lower', '_z', 'P_out_oxy_cath'] = 0.0
    mpc.bounds['lower', '_z', 'P_out_hyd_ano'] = 0.0
    mpc.bounds['lower', '_z', 'P_out_hyd_cath'] = 0.0
    mpc.bounds['lower', '_z', 'C_mix_oxy'] = 0.0
    mpc.bounds['lower', '_z', 'C_mix_hyd'] = 0.0
    mpc.bounds['lower', '_z', 'V_dot_g_ano'] = 0.0
    mpc.bounds['lower', '_z', 'V_dot_g_cath'] = 0.0
    mpc.bounds['lower', '_z', 'C_out_oxy_ano'] = 0.0
    mpc.bounds['lower', '_z', 'C_out_oxy_cath'] = 0.0
    mpc.bounds['lower', '_z', 'C_out_hyd_ano'] = 0.0
    mpc.bounds['lower', '_z', 'C_out_hyd_cath'] = 0.0
    mpc.bounds['lower', '_z', 'I'] = 0.0

    mpc.bounds['upper', '_x','P_hyd_tank'] = 400*101325
    mpc.bounds['upper', '_x','E_battery'] = 0.9*B_cap*3600

    # Nonlinear constraints:

    mpc.set_nl_cons('P_balance_bounded',(model.u['Power_elect']+model.u['Power_charg'])-model.tvp['Power_solar'],0)
    mpc.set_nl_cons('H2_impurity_ub', (model.z['P_out_hyd_ano']/(p_ano-p_water))-0.02,0)
    mpc.set_nl_cons('Current_density_ub', (I/A_el)-5000, 0)

    # Input constraints:
    mpc.bounds['lower','_u','Power_disch'] = 0
    mpc.bounds['upper','_u','Power_disch'] = (1*B_cap*Volt_disch)
    mpc.bounds['lower','_u','Power_elect'] = 0
    mpc.bounds['lower','_u','Power_charg'] = 0
    mpc.bounds['upper','_u','Power_charg'] = 1*B_cap*Volt_charg
    
    ########## Objective function ##########

    mterm = (1e5*((model.x['P_hyd_tank']/101325)-200)**2) 
    lterm = (1e4*((model.z['I']/A_el)-2720)**2)+(1e5*((model.x['P_hyd_tank']/101325)-200)**2)+1e-6*((model.tvp['Power_solar'])-(model.u['Power_elect']+model.u['Power_charg']))**2
    lterm += 1e1*((model.u['Power_disch']*model.u['Power_charg'])-1e-3)**2
    
    mpc.set_objective(mterm=mterm,lterm=lterm)
    
    
    mpc.set_rterm(Power_charg=10)
    mpc.set_rterm(Power_elect=10)
    
    ########## Scaling ##########

    mpc.scaling['_z', 'C_out_oxy_ano'] = 1e-2
    mpc.scaling['_z', 'C_out_oxy_cath'] = 1e-4
    mpc.scaling['_z', 'C_out_hyd_ano'] = 1e-4
    mpc.scaling['_z', 'C_out_hyd_cath'] = 1e-1
    mpc.scaling['_z', 'P_out_oxy_ano'] = 1e4
    mpc.scaling['_z', 'P_out_oxy_cath'] = 1e2
    mpc.scaling['_z', 'V_dot_g_ano'] = 1e-3
    mpc.scaling['_z', 'V_dot_g_cath'] = 1e-3
    mpc.scaling['_z', 'C_mix_oxy'] = 1e-2
    mpc.scaling['_z', 'C_mix_hyd'] = 1e-2
    mpc.scaling['_z', 'P_out_hyd_cath'] = 1e4
    mpc.scaling['_z', 'P_out_hyd_ano'] = 1e2
    mpc.scaling['_z', 'I'] = 1e4
    

    mpc.scaling['_x', 'P_hyd_tank'] = 1e7
    mpc.scaling['_x', 'E_battery'] = 1e10  #1e8

    mpc.scaling['_u', 'Power_disch'] = 1e5
    mpc.scaling['_u', 'Power_elect'] = 1e5
    mpc.scaling['_u', 'Power_charg'] = 1e7
 

    ########## Uncertinity and tvp values ##########

    tvp_num_mpc = mpc.get_tvp_template()
    
    def tvp_fun_mpc(t_now):
        
        for i in range(setup_mpc['n_horizon']):
            tvp_num_mpc['_tvp',i] = np.array([solar_power_baseline[(int(((t_now)/3600)))+i],demand_baseline[(int(((t_now)/3600)))+i]])
        
        return tvp_num_mpc

    mpc.set_tvp_fun(tvp_fun_mpc)

    mpc.setup()

    return mpc

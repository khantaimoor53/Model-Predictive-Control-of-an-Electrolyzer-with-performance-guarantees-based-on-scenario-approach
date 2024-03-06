# %%
import pandas as pd
import numpy as np

def demand_700():

    ### Create an array for hourly demand profile

    df = pd.read_csv('Demand_Profile_700.csv', index_col=0)
    df_names = df.columns
    df_demand = df['Demand ']
    demand = df_demand.to_numpy()
    demand = demand*(1/(3600*2*1e-3))   # convert kg/h to mol/s (demand without uncertinity)
    demand = np.tile(demand, 2)         # repeat the demand for 2 weeks

    demand_unc = np.zeros(len(demand))   # introduce uncertainty in the demand
    for i in range(len(demand)):
        
        if 1 == 2:        #i >= 125 and i <= 174  # only use if you want to change range of uncertinity during week
            lower = demand[i]*0.65
            upper = demand[i]*1.1
            value_demand_unc = np.random.uniform(lower, upper)
            demand_unc[i] = value_demand_unc
        else:
            lower = demand[i]*0.6
            upper = demand[i]*1.6
            value_demand_unc = np.random.uniform(lower, upper)
            demand_unc[i] = value_demand_unc

    ### Create an array for hourly solar power profile depending on the irradiation and panel area

    df_irradiation = pd.read_csv('Solar_irradiation.csv', index_col=0)
    df_irradiation = df_irradiation['ALLSKY_SFC_SW_DWN']
    irradiation = df_irradiation.to_numpy()       # irradiation without uncertainty
    irradiation_unc = np.zeros(len(irradiation))  # introduce uncertainty in the irradiation

    for i in range(len(irradiation)):
        lower = irradiation[i]*0.8
        upper = irradiation[i]*1.2
        value = np.random.uniform(lower, upper)
        irradiation_unc[i] = value
    
    irradiation_sample_unc = irradiation_unc[0:(24*14)]  # 2 weeks of irradiation with uncertinity	
    irradiation_sample = irradiation[0:(24*14)]          # 2 weeks of irradiation without uncertinity
    panel_area = 4000*20  # previous 4300*20
    solar_power_sample_unc = irradiation_sample_unc * panel_area*0.22   # convert irradiation to power
    solar_power_sample = irradiation_sample * panel_area*0.22           # convert irradiation to power                 
   
    return solar_power_sample_unc, solar_power_sample, demand, demand_unc, irradiation_sample, panel_area


# %%

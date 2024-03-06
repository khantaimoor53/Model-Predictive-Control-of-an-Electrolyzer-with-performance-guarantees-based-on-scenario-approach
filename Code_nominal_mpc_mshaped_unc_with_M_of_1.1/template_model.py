# %%
import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc
# %%
def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------------
    template_model: Variables / RHS / ALG/ AUX/ Parameters/ Time-varying parameters
    --------------------------------------------------------------------------------
    """

    # Define model type:
    model_type = 'continuous'
    model=do_mpc.model.Model(model_type)

    # Define states:

    # Differential states
    P_hyd_tank = model.set_variable('_x', 'P_hyd_tank')             # pressure in storage tank [pa]  
    E_battery = model.set_variable('_x', 'E_battery')               # charge in battery [Coulomb]

    # Algebraic states
    C_out_oxy_ano = model.set_variable('_z', 'C_out_oxy_ano')
    C_out_oxy_cath = model.set_variable('_z', 'C_out_oxy_cath')
    C_out_hyd_ano = model.set_variable('_z', 'C_out_hyd_ano')
    C_out_hyd_cath = model.set_variable('_z', 'C_out_hyd_cath')
    P_out_hyd_ano = model.set_variable('_z', 'P_out_hyd_ano')
    P_out_hyd_cath = model.set_variable('_z', 'P_out_hyd_cath')
    V_dot_g_ano = model.set_variable('_z', 'V_dot_g_ano')
    V_dot_g_cath = model.set_variable('_z', 'V_dot_g_cath')
    C_mix_oxy = model.set_variable('_z', 'C_mix_oxy')
    C_mix_hyd = model.set_variable('_z', 'C_mix_hyd')
    P_out_oxy_ano = model.set_variable('_z', 'P_out_oxy_ano')
    P_out_oxy_cath = model.set_variable('_z', 'P_out_oxy_cath')
    I = model.set_variable('_z', 'I')                                # current flowing in a single cell [A]

    # Define inputs:
    Power_elect = model.set_variable('_u',  'Power_elect')           # amount of solar power given to the electrolyzer [W] 
    Power_charg = model.set_variable('_u',  'Power_charg')           # amount of solar power given to the battery for charging [W]
    Power_disch = model.set_variable('_u',  'Power_disch')           # amount of power discharged from battery towards electrolyzer [W]
    
    #Uncertian parameters:
    

    #Time-varying parameters/uncertian parameters:

    Power_solar = model.set_variable('_tvp', 'Power_solar')
    n_dot_demand = model.set_variable('_tvp', 'n_dot_demand')

    # Certian parameters:
    V_tank = 0.3*14            # 14 tanks of 300 liters with max 500 bar limit [m^3]

    
    N_cell = 20                # denotes the number of cells in a stack
    rho_hyd = 0.08375          # density of hydrogen at STP [kg/m^3]
    M_hyd = 1.00784*10**-3     # molar mass of hydrogen [kg/mol]
    M_oxy = 15.999*10**-3      # mplar mass of oxygen [kg/mol]
    z = 2                      # number of electrons involved in the reaction
    A_el = 2.66                # area of electrode in a cell [m^2]
    d_sep = 500*10**-6         # thickness of seperating membrane [m]
    eta = 0.5                  # porosity of seperating membrane
    F = 96485.3321             # Faraday's constant [C/mol] 
    R = 8.314                                           
    v_hyd_cath = 1             # stoichiometric coefficient
    v_oxy_ano = 0.5
    tau = 3.14                 # tortuosity
    w_KOH = 0.3                # mass fraction of KOH in the electrolyte
    k_hyd = 3.14   
    k_oxy = 3.66          
    T = 333.15                 # Average operating temperature of electrolyzer [K]
    H_hyd = 6.95774941*10**4   # henry's constant[atm ] converted to pa in eq
    H_oxy = 4.35252731*10**4   # henry's constant[atm ]
    M = 0.018                 
    theta = T - 273.15         # Average operating temperature of electrolyzer[degrees]
    V_dot_l_ano = 5*10**-4     # Flow rate of circulating electrolyte in anodic compartment of one cell [m^3/s]
    V_dot_l_cath = 5*10**-4    # Flow rate of circulating electrolyte in cathodic compartment of one cell [m^3/s]
    A_sep = 2.66               # Surface area of the seperating membrane [m^2]                                          
    v_hcell = 0.0126           # Volume of a single half cell [m^3]
    p_not = 101325             # average operating pressure of electrolyzer (un-pressurized operation) [pa]
    r1 = 8.05e-5               # parameter for polarization curve of electrolyzer
    r2 = -2.5e-7
    s = 0.185
    t1 = 1.002
    t2 = 8.424
    t3 = 247.3
    f1 = 200
    f2 = 0.985

    B_cap = 144000*20          # battery capacity required for 20h of operation[Ah]    #135*3600*5 
    Volt_charg = 4.2           # average charging voltage of a Li-ion battery [volts]
    Volt_disch = 3.7           # average discharging voltage of a Li-ion battery [volts]
    
    # Auxiliary equations:
    J = I/(A_el)
    eta_f = (((J*1e-1)**2)*f2)/(f1+(J*1e-1)**2)
    n_dot_r_hyd_cath = eta_f*(v_hyd_cath*J*A_el)/(z*F)   
    n_dot_r_hyd_ano = 0
    n_dot_r_oxy_ano = eta_f*(v_oxy_ano*J*A_el)/(z*F)  
    n_dot_r_oxy_cath = 0
    D_hyd_k = (8.04542e-9*(w_KOH**0))+(-2.07309e-8*(w_KOH**1))+(2.02214e-8*(w_KOH**2))   # caution! coefficents are temperature dependent
    D_oxy_k = (4.27612e-9*(w_KOH**0))+(-1.90911e-8*(w_KOH**1))+(3.6684e-8*(w_KOH**2))+(-2.53386e-8*(w_KOH**3))
    D_hyd_k_eff = D_hyd_k*(eta/tau)
    D_oxy_k_eff = D_oxy_k*(eta/tau)
    N_cross_hyd = (D_hyd_k_eff/d_sep)*(C_out_hyd_cath-C_out_hyd_ano)
    N_cross_oxy = (D_oxy_k_eff/d_sep)*(C_out_oxy_cath-C_out_oxy_ano)
    fg_hyd = 0.25744*(J**0.14134)
    fg_oxy = 1 #0.25744*(J**0.14134)-0.01    #could be assumed as one as stated by Turek's paper or an approximation is used
    rho_water = (999.8395+(16.945176*theta)-((7.9870401*10**-3)*(theta**2))-(46.170461*10**-6)*(theta**3)+(105.56302*10**-9)*(theta**4)-(280.54253*10**-12)*(theta**5))/(1+((16.879850*10**-3)*theta))
    C_eq_hyd_cath_wat = (rho_water*P_out_hyd_cath)/(M*101325*H_hyd)
    C_eq_oxy_cath_wat = (rho_water*P_out_oxy_cath)/(M*101325*H_oxy)
    C_eq_hyd_ano_wat = (rho_water*P_out_hyd_ano)/(M*101325*H_hyd)
    C_eq_oxy_ano_wat = (rho_water*P_out_oxy_ano)/(M*101325*H_oxy)
    C_eq_hyd_cath = C_eq_hyd_cath_wat/(10**(w_KOH*k_hyd))
    C_eq_oxy_cath = C_eq_oxy_cath_wat/(10**(w_KOH*k_oxy))
    C_eq_hyd_ano = C_eq_hyd_ano_wat/(10**(w_KOH*k_hyd))
    C_eq_oxy_ano = C_eq_oxy_ano_wat/(10**(w_KOH*k_oxy))
    d_b_cath = 593.84*((1+0.2*J)**-0.25)*10**-6          # here in meters
    d_b_ano = 100*10**-6 #((593.84*((1+0.2*J)**-0.25))-4*10**-5)*10**-6   # value reaches a steady size at practical range of current densities
    epsilon_g_out_ano = 0.59438-0.59231*(0.75647**(J*10**-3))
    epsilon_g_out_cath = 0.76764-0.73233*(0.73457**(J*10**-3))
    epsilon_g_cath = V_dot_g_cath/v_hcell
    epsilon_g_ano = V_dot_g_ano/v_hcell
    rho_sol = ((1001.53053*theta**0)+(-0.08343*theta**1)+(-0.00401*theta**2)+(5.51232e-6*(theta**3))+(-8.20994e-10*(theta**4)))*exp(0.86*w_KOH)
    eta = (0.9105535967*T**0)+(-0.01062211683*T**1)+(4.680761561e-5*(T**2))+(-9.209312883e-8*(T**3))+(6.814919843e-11*(T**4))
    u_b_cath = 0.33*(9.8**0.76)*((rho_sol/eta)**0.52)*((d_b_cath/2)**1.28)
    u_b_ano = 0.33*(9.8**0.76)*((rho_sol/eta)**0.52)*((d_b_ano/2)**1.28)
    u_sw_cath = (u_b_cath*(1-epsilon_g_cath))/(1+(epsilon_g_cath/((1-epsilon_g_cath)**2))+(1.05/(-0.5+((1+(0.0685/(epsilon_g_cath**2)))**0.5)))) # bubble rise velocity could be used instead to simplify calculations
    u_sw_ano = (u_b_ano*(1-epsilon_g_ano))/(1+(epsilon_g_ano/((1-epsilon_g_ano)**2))+(1.05/(-0.5+((1+(0.0685/(epsilon_g_ano**2)))**0.5))))
    Re_cath = (rho_sol*d_b_cath*u_b_cath)/eta
    Re_ano = (rho_sol*d_b_ano*u_b_ano)/eta
    Sc_hyd = eta/(rho_sol*D_hyd_k)
    Sc_oxy = eta/(rho_sol*D_oxy_k)
    k_l_hyd_cath = (2+((0.651*(Re_cath*Sc_hyd)**1.72)/(1+(Re_cath*Sc_hyd)**1.22)))*(D_hyd_k/d_b_cath)
    k_l_hyd_ano = (2+((0.651*(Re_ano*Sc_hyd)**1.72)/(1+(Re_ano*Sc_hyd)**1.22)))*(D_hyd_k/d_b_ano)
    k_l_oxy_cath = (2+((0.651*(Re_cath*Sc_oxy)**1.72)/(1+(Re_cath*Sc_oxy)**1.22)))*(D_oxy_k/d_b_cath)
    k_l_oxy_ano = (2+((0.651*(Re_ano*Sc_oxy)**1.72)/(1+(Re_ano*Sc_oxy)**1.22)))*(D_oxy_k/d_b_ano)

    N_phys_hyd_cath = k_l_hyd_cath*(C_eq_hyd_cath-C_out_hyd_cath)   
    N_phys_hyd_ano = k_l_hyd_ano*(C_eq_hyd_ano-C_out_hyd_ano)
    N_phys_oxy_cath = k_l_oxy_cath*(C_eq_oxy_cath-C_out_oxy_cath)
    N_phys_oxy_ano = k_l_oxy_ano*(C_eq_oxy_ano-C_out_oxy_ano)
    V_b_cath = (3.14/6)*(d_b_cath**3)
    V_b_ano = (3.14/6)*(d_b_ano**3)
    S_b_cath = 3.14*(d_b_cath**2)
    S_b_ano = 3.14*(d_b_ano**2)
    mKOH = w_KOH/(0.0561*(1-w_KOH))
    log_p_water = (-0.01508 * mKOH )- (0.0016788 * (mKOH**2)) + (2.25887e-5 * (mKOH**3)) + (1 -( 0.0012062 * mKOH) + (5.6024e-4 * (mKOH**2)) -( 7.8228e-6 * (mKOH**3))) * (35.4462 - (3343.93/T) - (10.9 * log10(T)) + (0.0041645 * T))
    p_water = (10**(log_p_water))*10**5  # pressure in bar converted to pa. so eq is diff !!!
    gamma = (0.065887)+(0.024546*w_KOH)+(0.131952*w_KOH**2)+(-0.033064*w_KOH**3)
    delta_p_ano = (4*gamma)/(d_b_ano)           # [N/m^2]
    delta_p_cath = (4*gamma)/(d_b_cath)         # [N/m^2]
    p_ano = p_not + delta_p_ano                 # [N/m^2]
    p_cath = p_not + delta_p_cath               # [N/m^2]
    V_gas_ano = epsilon_g_out_ano*v_hcell*(p_not/p_ano)
    V_gas_cath = epsilon_g_out_cath*v_hcell*(p_not/p_cath)
    A_gl_cath = V_gas_ano*S_b_cath/V_b_cath
    A_gl_ano = V_gas_cath*S_b_ano/V_b_ano

    #Algebraic and differential equations:
    model.set_rhs('P_hyd_tank', ((V_dot_g_cath*N_cell*P_out_hyd_cath*(25+273))/(V_tank*T))-((R*(25+273)*((n_dot_demand)))/V_tank))
    model.set_rhs('E_battery', (Power_charg/Volt_charg)-(Power_disch/Volt_disch))
    model.set_alg('C_out_oxy_ano', V_dot_l_ano*(C_mix_oxy-C_out_oxy_ano)+(N_phys_oxy_ano*A_gl_ano)+N_cross_oxy*A_sep+((1-fg_oxy)*n_dot_r_oxy_ano))
    model.set_alg('C_out_oxy_cath', V_dot_l_cath*(C_mix_oxy-C_out_oxy_cath)+(N_phys_oxy_cath*A_gl_cath)-N_cross_oxy*A_sep+((1-fg_oxy)*n_dot_r_oxy_cath))
    model.set_alg('C_out_hyd_ano', V_dot_l_ano*(C_mix_hyd-C_out_hyd_ano)+N_phys_hyd_ano*A_gl_ano+N_cross_hyd*A_sep+(1-fg_hyd)*n_dot_r_hyd_ano)
    model.set_alg('C_out_hyd_cath', V_dot_l_cath*(C_mix_hyd-C_out_hyd_cath)+N_phys_hyd_cath*A_gl_cath-N_cross_hyd*A_sep+(1-fg_hyd)*n_dot_r_hyd_cath)
    model.set_alg('P_out_hyd_ano',((V_dot_g_ano*(-P_out_hyd_ano))/(R*T))-N_phys_hyd_ano*A_gl_ano+fg_hyd*n_dot_r_hyd_ano)
    model.set_alg('P_out_hyd_cath',((V_dot_g_cath*(-P_out_hyd_cath))/(R*T))-N_phys_hyd_cath*A_gl_cath+fg_hyd*n_dot_r_hyd_cath)
    model.set_alg('V_dot_g_ano',((V_dot_g_ano*(-P_out_oxy_ano))/(R*T))-N_phys_oxy_ano*A_gl_ano+fg_oxy*n_dot_r_oxy_ano)
    model.set_alg('V_dot_g_cath',((V_dot_g_cath*(-P_out_oxy_cath))/(R*T))-N_phys_oxy_cath*A_gl_cath+fg_oxy*n_dot_r_oxy_cath)
    model.set_alg('C_mix_oxy', (V_dot_l_ano*C_out_oxy_ano+V_dot_l_cath*C_out_oxy_cath-(V_dot_l_ano+V_dot_l_cath)*C_mix_oxy))
    model.set_alg('C_mix_hyd', (V_dot_l_ano*C_out_hyd_ano+V_dot_l_cath*C_out_hyd_cath-(V_dot_l_ano+V_dot_l_cath)*C_mix_hyd))
    model.set_alg('P_out_oxy_ano', p_ano-P_out_oxy_ano-P_out_hyd_ano-p_water)
    model.set_alg('P_out_oxy_cath', p_cath-P_out_oxy_cath-P_out_hyd_cath-p_water)
    model.set_alg('I', ((1.2+((r1+r2*theta)/A_el)*I+s*log10(((t1+(t2/theta)+(t3/theta**2))*((I)/A_el))+1))*I)-((Power_elect + Power_disch)/N_cell))

    #Auxiliary equations for extracting data:
    model.set_expression('Power_disca', Power_solar- Power_elect - Power_charg)	
    model.set_expression('H2_impurity', (P_out_hyd_ano/(p_ano-p_water))*100)
    model.set_expression('SOC',((E_battery/(B_cap*3600)))*100 )
    model.set_expression('Total_Power_elect', Power_disch + Power_elect)
    model.set_expression('Power_solar', Power_solar)
    model.set_expression('Current_density', I/A_el)
    model.set_expression('mol_produced',((V_dot_g_cath*P_out_hyd_cath)/(R*T)))
    model.set_expression('Volt',(1.2+((r1+r2*theta)/A_el)*I+s*log10(((t1+(t2/theta)+(t3/theta**2))*((I)/A_el))+1)))
    model.set_expression('Efficency', (((J*1e-1)**2)*f2)/(f1+(J*1e-1)**2))
    
    model.setup()

    return model, B_cap, p_ano, p_water, I, A_el, Volt_disch, Volt_charg, N_cell
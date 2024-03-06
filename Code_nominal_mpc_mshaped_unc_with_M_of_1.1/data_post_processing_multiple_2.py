# %%
from do_mpc import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

######### Data Processing #########
start_time = time.time()

plan_1 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_1_result/plan1.pkl')
dh = sampling.DataHandler(plan_1)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_1_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res1 = dh[:]

plan_2 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_2_result/plan2.pkl')
dh = sampling.DataHandler(plan_2)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_2_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res2 = dh[:]

plan_3 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_3_result/plan3.pkl')
dh = sampling.DataHandler(plan_3)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_3_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res3 = dh[:]

plan_4 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_4_result/plan4.pkl')
dh = sampling.DataHandler(plan_4)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_4_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res4 = dh[:]

plan_5 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_5_result/plan5.pkl')
dh = sampling.DataHandler(plan_5)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_5_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res5 = dh[:]

plan_6 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_6_result/plan6.pkl')
dh = sampling.DataHandler(plan_6)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_6_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res6 = dh[:]

plan_7 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_7_result/plan7.pkl')
dh = sampling.DataHandler(plan_7)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_7_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res7 = dh[:]

plan_8 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_8_result/plan8.pkl')
dh = sampling.DataHandler(plan_8)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_8_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res8 = dh[:]

plan_9 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_9_result/plan9.pkl')
dh = sampling.DataHandler(plan_9)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_9_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res9 = dh[:]

plan_10 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_10_result/plan10.pkl')
dh = sampling.DataHandler(plan_10)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_10_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res10 = dh[:]

plan_11 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_11_result/plan11.pkl')
dh = sampling.DataHandler(plan_11)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_11_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res11 = dh[:]

plan_12 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_12_result/plan12.pkl')
dh = sampling.DataHandler(plan_12)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_12_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res12 = dh[:]

plan_13 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_13_result/plan13.pkl')
dh = sampling.DataHandler(plan_13)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_13_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res13 = dh[:]

plan_14 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_14_result/plan14.pkl')
dh = sampling.DataHandler(plan_14)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_14_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res14 = dh[:]

plan_15 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_15_result/plan15.pkl')
dh = sampling.DataHandler(plan_15)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_15_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res15 = dh[:]

plan_16 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_16_result/plan16.pkl')
dh = sampling.DataHandler(plan_16)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_16_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])

res16 = dh[:]

plan_17 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_17_result/plan17.pkl')
dh = sampling.DataHandler(plan_17)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_17_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res17 = dh[:]

plan_18 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_18_result/plan18.pkl')
dh = sampling.DataHandler(plan_18)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_18_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res18 = dh[:]

plan_19 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_19_result/plan19.pkl')
dh = sampling.DataHandler(plan_19)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_19_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res19 = dh[:]

plan_20 = tools.load_pickle('./sampling_results_history/LiDo_results/20_360_runs/node_20_result/plan20.pkl')
dh = sampling.DataHandler(plan_20)
dh.data_dir = './sampling_results_history/LiDo_results/20_360_runs/node_20_result/'
dh.set_param(sample_name = 'sample')

dh.set_post_processing('inputs', lambda data: data[0]['_u'])
dh.set_post_processing('dyn_states', lambda data: data[0]['_x'])
dh.set_post_processing('alg_states', lambda data: data[0]['_z'])
dh.set_post_processing('tvp', lambda data: data[0]['_tvp'])
dh.set_post_processing('success', lambda data: data[1]['success'])
dh.set_post_processing('time', lambda data: data[0]['_time'])
dh.set_post_processing('outputs_y', lambda data: data[0]['_y'])
dh.set_post_processing('auxillary', lambda data: data[0]['_aux'])


res20 = dh[:]


end_time = time.time()
duration = end_time - start_time
print('Time taken for data processing: ', duration)

# %%
res_global =  res1 + res2 + res3 + res4 +res5+ res6 +res7 +res8 +res9 +res10 # join all result lists
res_global += res11 + res12 + res13 + res14 +res15+ res16 +res17 +res18 +res19 +res20
n_samples = 100 # cut size
n_cut = int((len(res_global))/n_samples) # number of cuts
# %%
#n_samples = int((len(plan_1)+len(plan_2)+len(plan_3)+len(plan_4)+len(plan_5)+len(plan_6)+len(plan_7)+len(plan_8)+len(plan_9)+len(plan_10))) # number of sample groups based on cut(22 of size 100)
#n_samples = int((n_samples+len(plan_11)+len(plan_12)+len(plan_13)+len(plan_14)+len(plan_15)+len(plan_17)+len(plan_18)+len(plan_19)+len(plan_20))/n_cut) # number of sample groups based on cut(22 of size 100)
 
n_sim_steps = len(res6[0]['inputs'])
n_indicators = 1
refrence = 63           # refrence cut for selecting max value (l)

# %%
def data_extraction(start,stop):
    
    res = res_global[start:stop]   # select the no. of samples to be processed

    ######### Calculate performance indicators #########


    cl_cost = np.zeros(n_samples)       # closed loop cost
    success_per = np.zeros(n_samples)   # success percentage
    simultaneous_ch_disch = np.zeros(n_samples) # simultaneous charging and discharging
    const_viol_avg = np.zeros(n_samples) # average constraint violation
    const_viol_max = np.zeros(n_samples) # maximum constraint violation

    # Closed loop cost:

    for i in range(n_samples):
        for j in range(n_sim_steps):
            cl_cost[i] += (1e4*((res[i]['alg_states'][j,12]/2.66)-4500)**2)
            cl_cost[i] += (1e-5*((res[i]['dyn_states'][j,0]/101325)-250)**2)
            cl_cost[i] += 10*((res[i]['tvp'][j,0])-(res[i]['inputs'][j,0]+res[i]['inputs'][j,1])*0.01*res[i]['tvp'][j,0])**2
            cl_cost[i] += 1e3*(res[i]['inputs'][j,2]*res[i]['inputs'][j,1]-10**-3)**2   
        
    
    # Success percentage:

    for i in range(n_samples):
        success_per[i] = sum(res[i]['success'])/n_sim_steps

    
    # Simultaneous charging and discharging:

    # for i in range(n_samples):
    #     for j in range(n_sim_steps):
    #         if res[i]['inputs'][j,1]*0.01*res[i]['tvp'][j,0] > 1 and res[i]['inputs'][j,2]*0.01*res[i]['tvp'][j,0] > 1:
    #             simultaneous_ch_disch[i] += 1

    for i in range(n_samples):
        for j in range(n_sim_steps):
            if res[i]['inputs'][j,1] > 0.1 and res[i]['inputs'][j,2] > 0.1:
                simultaneous_ch_disch[i] += 1
    
    # Battery average constraint violation

    for i in range(n_samples):
        for j in range(n_sim_steps):
            if res[i]['auxillary'][j,3]>90:
                const_viol_avg[i] += res[i]['auxillary'][j,3]-90
            if res[i]['auxillary'][j,3]<20:
                const_viol_avg[i] += 20-res[i]['auxillary'][j,3]
        const_viol_avg[i] = const_viol_avg[i]/n_sim_steps

    # Battery max constraint violation

    for i in range(n_samples):
        container = [0]
        for j in range(n_sim_steps):
            if res[i]['auxillary'][j,3]>90:
                container.append(res[i]['auxillary'][j,3]-90)
            if res[i]['auxillary'][j,3]<20:
                container.append(20-res[i]['auxillary'][j,3])
        const_viol_max[i] = max(container)


    return cl_cost, success_per, simultaneous_ch_disch, const_viol_avg, const_viol_max

def plot_cl_cost(cl_cost):
    
        plt.hist(cl_cost, bins=20)
        plt.xlabel('Closed loop cost')
        plt.ylabel('Frequency')
def plot_success_per(success_per):
    
        plt.hist(success_per, bins=20)
        plt.xlabel('Success percentage')
        plt.ylabel('Frequency')

def plot_simultaneous_ch_disch(simultaneous_ch_disch):
    
        plt.hist(simultaneous_ch_disch, bins=20)
        plt.xlabel('Simultaneous charging and discharging')
        plt.ylabel('Frequency')

def plot_const_viol_avg(const_viol_avg):
    
        plt.hist(const_viol_avg, bins=20)
        plt.xlabel('Average constraint violation')
        plt.ylabel('Frequency')
    
def plot_const_viol_max(const_viol_max):
        
            plt.hist(const_viol_max, bins=20)
            plt.xlabel('Maximum constraint violation')
            plt.ylabel('Frequency')
# %%
def viol_count_cl_cost():
    cl_cost_limit,_,_,_,_ = data_extraction(refrence*n_samples,(refrence+1)*n_samples)
    #l = max(cl_cost_limit)
    l = 1.58*10**24
    print('The maximum closed loop cost is: '+str(l))
    violations_cl_cost = np.zeros(n_cut) 
    for i in range(n_cut):
        
        start = i*n_samples
        stop = (i+1)*n_samples
        cl_cost, success_per, simultaneous_ch_disch, const_viol_avg,_ = data_extraction(start,stop)
        for j in range(len(cl_cost)):
            if cl_cost[j]>l:
                violations_cl_cost[i] += 1
    return violations_cl_cost

def viol_prob_cl_cost(violations_cl_cost, refrence):
    violation_prob_cl_cost = violations_cl_cost/n_samples
    violation_prob_cl_cost = np.delete(violation_prob_cl_cost, refrence, axis=0)
    return violation_prob_cl_cost

def plot_viol_prob_cl_cost():

    violations_cl_cost = viol_count_cl_cost()
    violation_prob_cl_cost = viol_prob_cl_cost(violations_cl_cost, refrence)
    fig, ax = plt.subplots(1,1)
    ax.hist(violation_prob_cl_cost,density=True, bins=7)
    ax.axvline(x=(sum(violation_prob_cl_cost)/len(violation_prob_cl_cost)), color='g', linestyle='--', label='Mean observed violation probability')
    ax.axvline(x=(n_indicators/(n_samples+1)), color='orange', linestyle='--', label='Mean calculated violation probability')
    ax.axvline(x = 0.1, color='black', linestyle='--', label='Violation probability bound ($ϵ$)')
    ax.set_xlabel('Violation probability ($V_p$)')
    ax.set_ylabel('Probability density')
    ax.legend()
    ax.set_xlim(0,0.15)
    result = []
    epsilon = []
    for i in np.arange(0,1,0.0001):
    
        result.append(100*(1-i)**99)
        epsilon.append(i)
    ax.plot(epsilon,result, color='red', label='PDF defined by the scenario approach ($f_{SA}(ϵ)$)')
    ax.legend()
    plt.savefig('violation_prob_cl_cost_mshaped_high1_0.1.pdf')



def viol_count_batt_constr():

    _,_,_,batt_constr_limit,_ = data_extraction(refrence*n_samples,(refrence+1)*n_samples)
    l = max(batt_constr_limit)
    print('The maximum average battery constraint violation is: '+str(l))
    violations_batt_constr = np.zeros(n_cut) 
    for i in range(n_cut):
        
        start = i*n_samples
        stop = (i+1)*n_samples
        _, _, _, batt_constr,_ = data_extraction(start,stop)
        for j in range(len(batt_constr)):
            if batt_constr[j] > l:
                violations_batt_constr[i] += 1
    return violations_batt_constr

def viol_prob_batt_constr(violations_batt_constr, refrence):
    violation_prob_batt_constr = violations_batt_constr/n_samples
    violation_prob_batt_constr = np.delete(violation_prob_batt_constr, refrence, axis=0)
    return violation_prob_batt_constr

def plot_viol_prob_batt_constr():

    violations_batt_constr = viol_count_batt_constr()
    violation_prob_batt_constr = viol_prob_batt_constr(violations_batt_constr, refrence)
    fig, ax = plt.subplots(1,1)
    ax.hist(violation_prob_batt_constr,density=True, bins=5)
    ax.axvline(x=(sum(violation_prob_batt_constr)/len(violation_prob_batt_constr)), color='g', linestyle='--', label='Mean observed violation probability')
    ax.axvline(x=(n_indicators/(n_samples+1)), color='orange', linestyle='--', label='Mean calculated violation probability')
    ax.axvline(x = 0.1, color='black', linestyle='--', label='Violation probability threshold ( $\epsilon$ )')
    ax.set_xlabel('Violation probability')
    ax.set_ylabel('Density')
    ax.set_xlim(0,0.15)

    result = []
    epsilon = []
    for i in np.arange(0,1,0.0001):
    
        result.append(100*(1-i)**99)
        epsilon.append(i)
    ax.plot(epsilon,result, color='red', label='SP')
    ax.legend()

def viol_count_batt_constr_max():
    _,_,_,_,batt_constr_max_limit = data_extraction(refrence*n_samples,(refrence+1)*n_samples)
    #l = max(batt_constr_max_limit)
    l = 2.01
    print('The maximum value of maximum battery constraint violation is: '+str(l))
    violations_batt_constr_max = np.zeros(n_cut) 
    for i in range(n_cut):
        
        start = i*n_samples
        stop = (i+1)*n_samples
        _, _, _,_, batt_constr_max = data_extraction(start,stop)
        for j in range(len(batt_constr_max)):
            if batt_constr_max[j] > l:
                violations_batt_constr_max[i] += 1
    return violations_batt_constr_max

def viol_prob_batt_constr_max(violations_batt_constr_max, refrence):
    violation_prob_batt_constr_max = violations_batt_constr_max/n_samples
    violation_prob_batt_constr_max = np.delete(violation_prob_batt_constr_max, refrence, axis=0)
    return violation_prob_batt_constr_max

def plot_viol_prob_batt_constr_max():
         
        violations_batt_constr_max = viol_count_batt_constr_max()
        violation_prob_batt_constr_max = viol_prob_batt_constr_max(violations_batt_constr_max, refrence)
        fig, ax = plt.subplots(1,1)
        ax.hist(violation_prob_batt_constr_max,density=True, bins=3)
        ax.axvline(x=(sum(violation_prob_batt_constr_max)/len(violation_prob_batt_constr_max)), color='g', linestyle='--', label='Mean observed violation probability')
        ax.axvline(x=(n_indicators/(n_samples+1)), color='orange', linestyle='--', label='Mean calculated violation probability')
        ax.axvline(x = 0.1, color='black', linestyle='--', label='Violation probability bound (ϵ)')
        ax.set_xlabel('Violation probability ($V_p$)')
        ax.set_ylabel('Probability density')
        ax.set_xlim(0,0.15)
    
        result = []
        epsilon = []
        for i in np.arange(0,1,0.0001):
        
            result.append(100*(1-i)**99)
            epsilon.append(i)
        ax.plot(epsilon,result, color='red', label='PDF defined by the scenario approach ($f_{SA}(ϵ)$)')
        ax.legend()
        plt.savefig('violation_prob_batt_constr_max_mshaped_high1_0.1.pdf')

def plot_viol_prob_dual():
     
    violations_cl_cost = viol_count_cl_cost()
    violation_prob_cl_cost = viol_prob_cl_cost(violations_cl_cost, refrence)
    violations_batt_constr = viol_count_batt_constr()
    violation_prob_batt_constr = viol_prob_batt_constr(violations_batt_constr, refrence)
    violation_prob_dual = violation_prob_cl_cost + violation_prob_batt_constr
    fig, ax = plt.subplots(1,1)
    ax.hist(violation_prob_dual,density=True, bins=6)
    ax.axvline(x=(sum(violation_prob_dual)/len(violation_prob_dual)), color='g', linestyle='--', label='Mean observed violation probability')
    ax.axvline(x=(2/(n_samples+1)), color='orange', linestyle='--', label='Mean calculated violation probability')
    ax.axvline(x = 0.1, color='black', linestyle='--', label='Violation probability threshold ($\epsilon$)')
    ax.set_xlabel('Violation probability')
    ax.set_ylabel('Density')
    ax.set_xlim(0,0.15)

    result = []
    epsilon = []
    for i in np.arange(0,1,0.0001):
    
        result.append(100*99*i*(1-i)**98)
        epsilon.append(i)
    ax.plot(epsilon,result, color='red', label='SP')
    ax.legend()

def plot_viol_prob_dual_2():
     
    violations_cl_cost = viol_count_cl_cost()
    violation_prob_cl_cost = viol_prob_cl_cost(violations_cl_cost, refrence)
    # violations_batt_constr = viol_count_batt_constr()
    # violation_prob_batt_constr = viol_prob_batt_constr(violations_batt_constr, refrence)
    violations_batt_constr_max = viol_count_batt_constr_max()
    violation_prob_batt_constr_max = viol_prob_batt_constr_max(violations_batt_constr_max, refrence)
    violation_prob_dual = violation_prob_cl_cost + violation_prob_batt_constr_max
    fig, ax = plt.subplots(1,1)
    ax.hist(violation_prob_dual,density=True, bins=7)
    ax.axvline(x=(sum(violation_prob_dual)/len(violation_prob_dual)), color='g', linestyle='--', label='Mean observed violation probability')
    ax.axvline(x=(2/(n_samples+1)), color='orange', linestyle='--', label='Mean calculated violation probability')
    ax.axvline(x = 0.1, color='black', linestyle='--', label='Violation probability bound ($ϵ$)')
    ax.set_xlabel('Violation probability ($V_p$)')
    ax.set_ylabel('Probability density')
    ax.set_xlim(0,0.15)

    result = []
    epsilon = []
    for i in np.arange(0,1,0.0001):
    
        result.append(100*99*i*(1-i)**98)
        epsilon.append(i)
    ax.plot(epsilon,result, color='red', label='PDF defined by the scenario approach ($f_{SA}(ϵ)$)')
    ax.legend()
    plt.savefig('violation_prob_dual_2_mshaped_high1_0.1.pdf')


# %%

The aim of this code is to give the reader an idea of how the results shown in the thesis are obtained. It should be noted that this is a basic representation of the code and multiple sampling.py files along with multiple batch scripts may need to be created to sample the data volume used in this study. Following are the functions of different parts of the code:

>template_model.py defines the model of the electrolyzer

>template_mpc.py defines the mpc controller

>template_simulator.py defines the simulater based on IDAS

>main.py can be used to call the above three scripts along with demand_700.py which provides the demand profiles and the power profiles to simulate the control loop and visualize it. This file is meant to run and visualize the MPC and not meant for sampling.

>sampling_plan.py can be used to generate scenarios based on uncertain power supply and demand. Please note that only uncertain power supply is considered in this study. 

>sampling.py can be used to run multiple instances of control-loop on multiple processor cores and obtain the data of control-loop simulation. This is the main code used to generate results for the scenario and ambiguous scenario approach. This script can be run on LiDo. If you wish to use more than one node, creating multiple scripts scuh as (sampling_1.py) and each script must load its distinct "plan" generated using sampling_plan.py and store the result in a seperate folder. You can do this by defining a distinct diractory for the results. Set the number of processes equal to number of processors that are available on a node. 

>data_post_processing_2.py has a function of processing results obtained from various nodes (in case LiDo is used) and to calculate the performance indicators

>A script defining an uncertainity distribution (parallel_mpc_1) in also added in case of non-uniform distribution as these distributions are not part of standard Python functions.

>The attached batch script can be used to create jobs for Slurm Workload Manager. A job in this case runs the sampling on only one node on a cluster computer. 
It can be configured to request desired resources for running the job. Multiple batch files can be created using the given tempelate to utilize more than one node. The resulting data can
be combined using data_post_processing_2.py

Following procedure can be used if the reader wants to run the sampling of the closed-loop using LiDo:

1) Generate sampling plans as .pkl files using the script: sampling_plan.py and store then in a seperate folder in same directory.
2) Generate copies of the script sampling.py in the same directory and configure each to reciveve a different plan.pkl file.
3) The number of copies should be equal to the number of nodes you want to use.
4) Each sampling.py file should store its results in a different folder.
4) Upload the directory to the LiDo and create equal copies of the provided batch file.
5) Each batch file describes a job that runs on one node.
6) Set the number of parallel processes in sampling.py to the maximum available cores on a single node to run the optimization problem on each core in parallel.
7) Approximate and set the required computation time and name an output folder for the output of the python code.
8) Upload and run the batch files using (sbatch) command in the linux shell of LiDo. 
9) After successful execution, download the folder containing .pkl files for the results and copy the plan.pkl file that was used to generate those results in the same folder.
10) Use data_post_processing_2.py to generate results for the scenario approach and the scenario approach with ambiguity.
  
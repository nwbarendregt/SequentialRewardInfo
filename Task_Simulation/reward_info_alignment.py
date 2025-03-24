"""
reward_info_alignment.py
Performs sweep over task parameter space (epsilon, q) using both RewardMax and InfoMax models and extracts their behavioral alignment.
Script designed to utilize multiprocessing for faster sweep speeds.
Data generated from script used to generate Fig. 3.
"""
import numpy as np
import sequential_methods as sm
from multiprocessing import Pool
import csv
# Define task and model parameters:
N = 10 # Time step budget
N_block = int(1e4) # Number of realizations per parameter set
epsilon = np.linspace(0, 0.5); q = np.linspace(0.5, 1) # Parameter space of switching rates and reward reliabilities
h = 0.75 # Bernoulli parameter for environmental evidence
R = (100, -100) # Reward structure (reward, punishment)
c = (1, 1) # Action time step costs (commitment, sampling)
gamma = np.array([0, 0.25, 0.5, 0.75, 1]) # Parameter space for future utility discounting
# Create multiprocessing interable list and index map for data exporting:
iterable = []
index_dict = {}
l = 0
for i in range(len(epsilon)):
    for j in range(len(q)):
        for k in range(len(gamma)):
            iterable.append((epsilon[i], q[j], gamma[k]))
            index_dict[l] = (i, j, k)
            l += 1
# Define alignment simulation helper function for multiprocessing:
def action_alignment(epsilon, q, gamma):
    # Generate task design object:
    task_design = sm.task.BernoulliInference(N, epsilon, q, R, c, h)
    # Generate agent's rewardmax model object:
    rm_model = sm.model.RewardMax(gamma)
    # Combine task and rewardmax model objects:
    rm_behavior = sm.experiment.Behavior(task_design,rm_model)
    # Generate agent's infomax model object:
    im_model = sm.model.InfoMax(gamma)
    # Combine task and infomax model objects:
    im_behavior = sm.experiment.Behavior(task_design, im_model)
    # Pre-allocate results vector:
    alignment = np.array([])
    for _ in range(N_block):
        # Simulate agent's rewardmax behavior in task realization:
        rm_results = sm.experiment.simulate(rm_behavior)
        # Simulate agent's infomax behavior in task realization:
        im_results = sm.experiment.simulate(im_behavior)
        # Construct state likelihood mesh:
        p_mesh = np.linspace(0, 1, num=len(rm_behavior.opt_action[:, 0]))
        # Convert LLR beliefs to state likelihood beliefs:
        rm_p = 1/(1+np.exp(-rm_results.belief))
        im_p = 1/(1+np.exp(-im_results.belief))
        # Pre-allocate reward-to-information alignment:
        r_alignment_i = 0
        # Pre-allocate information-to-reward alignment:
        i_alignment_i = 0
        for i in range(len(rm_p)):
            # If an infomax strategy takes the same action as the rewardmax strategy
            # given the rewardmax state belief:
            if ((rm_behavior.opt_action[np.argmin(np.abs(rm_p[i]-p_mesh)), i] 
                == im_behavior.opt_action[np.argmin(np.abs(rm_p[i]-p_mesh)), i]) 
                & ~np.isnan(rm_p[i])):
                r_alignment_i += 1 # Increment reward-to-information alignment
        for i in range(len(im_p)):
            # If a rewardmax strategy takes the same action as the infomax strategy
            # given the informax state belief:
            if ((rm_behavior.opt_action[np.argmin(np.abs(im_p[i]-p_mesh)), i] 
                == im_behavior.opt_action[np.argmin(np.abs(im_p[i]-p_mesh)), i]) 
                & ~np.isnan(im_p[i])):
                i_alignment_i += 1 # Increment information-to-reward alignment
        # Store alignment as the average between reward-to-information and information-to-reward alignments:
        alignment = np.append(alignment, (r_alignment_i/np.sum(~np.isnan(rm_p))+i_alignment_i/np.sum(~np.isnan(im_p)))/2)
    return np.mean(alignment)
# Perform alignment parameter sweep using multiprocessing:
with Pool(25) as p:
	results = np.asarray(p.starmap(action_alignment, iterable))
# Format and save parameter sweep data as csv:        
data_fields = ['epsilon', 'q', 'gamma', 'rho_mean']
data_rows = []
for i in range(len(index_dict)):
	data_rows.append([epsilon[index_dict[i][0]], q[index_dict[i][1]], gamma[index_dict[i][2]], results[i]])
with open('alignment_standard.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(data_fields)
    csvwriter.writerows(data_rows)

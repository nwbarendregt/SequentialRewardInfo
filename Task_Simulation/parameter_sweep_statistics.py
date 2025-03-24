"""
parameter_sweep_statistics.py
Performs sweep over task parameter space (epsilon, q) using both RewardMax and InfoMax models and extracts their performance and behavioral statistics.
Script designed to utilize multiprocessing for faster sweep speeds.
Data generated from script used to generate Fig. 2C, Fig. 2D, and Fig. 4.
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
gamma = np.array([0, 0.25, 0.5, 0.75, 1]) # Parameter space of future utility discounting
# Create multiprocessing iterable list and index map for data exporting:
iterable = []
index_dict = {}
l = 0
for i in range(len(epsilon)):
    for j in range(len(q)):
        for k in range(len(gamma)):
            iterable.append((epsilon[i], q[j], gamma[k]))
            index_dict[l] = (i, j, k)
            l += 1
# Define rewardmax simulation helper function for multiprocessing:
def reward_block_sim(epsilon, q, gamma):
    # Generate task design object:
    task_design = sm.task.BernoulliInference(N, epsilon, q, R, c, h)
    # Generate agent's rewardmax model object:
    model = sm.model.RewardMax(gamma)
    # Combine task and model objects:
    behavior = sm.experiment.Behavior(task_design, model)
    # Pre-allocate results vectors:
    rho = np.array([]) # Reward rate statistics
    IG = np.array([]) # Information gain statistics
    ICI = np.array([]) # Commit burst length statistics
    ISI = np.array([]) # Sample burst length statistics
    for _ in range(N_block):
        # Simulate agent's behavior in task realization:
        results = sm.experiment.simulate(behavior)
        # Store performance and behavioral statistics:
        rho = np.append(rho, sm.analysis.reward_rate(results))
        IG = np.append(IG, sm.analysis.information_gain(results))
        seq_len = sm.analysis.sequence_lengths(results)
        ICI = np.append(ICI, seq_len[0])
        ISI = np.append(ISI, seq_len[1])
    return np.mean(rho), np.std(rho), np.mean(IG), np.std(IG), np.mean(ICI), np.std(ICI), np.mean(ISI), np.std(ISI)
# Perform rewardmax parameter sweep using multiprocessing:
with Pool(25) as p:
	results = np.asarray(p.starmap(reward_block_sim, iterable))
# Format and save parameter sweep data as csv:
data_fields = ['epsilon', 'q', 'gamma', 'rho_mean', 'rho_std', 'IG_mean', 'IG_std', 'ICI_mean', 'ICI_std', 'ISI_mean', 'ISI_std']
data_rows = []
for i in range(len(index_dict)):
	data_rows.append([epsilon[index_dict[i][0]], q[index_dict[i][1]], gamma[index_dict[i][2]], results[i, 0], results[i, 1], results[i, 2], results[i, 3], results[i, 4], results[i, 5], results[i, 6], results[i, 7]])	
with open('rm_standard.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(data_fields)
    csvwriter.writerows(data_rows)
# Define infomax simulation helper function for multiprocessing:   
def info_block_sim(epsilon, q, gamma):
    # Generate task design object:
    task_design = sm.task.BernoulliInference(N, epsilon, q, R, c, h)
    # Generate agent's info model object:
    model = sm.model.InfoMax(gamma)
    # Combine task and model objects:
    behavior = sm.experiment.Behavior(task_design, model)
    # Pre-allocate results vectors:
    rho = np.array([]) # Reward rate statistics
    IG = np.array([]) # Information gain statistics
    ICI = np.array([]) # Commit burst length statistics
    ISI = np.array([]) # Sample burst length statistics
    for _ in range(N_block):
        # Simulate agent's behavior in task realization:
        results = sm.experiment.simulate(behavior)
        # Store performance and behavioral statistics:
        rho = np.append(rho, sm.analysis.reward_rate(results))
        IG = np.append(IG, sm.analysis.information_gain(results))
        seq_len = sm.analysis.sequence_lengths(results)
        ICI = np.append(ICI, seq_len[0])
        ISI = np.append(ISI, seq_len[1])
    return np.mean(rho), np.std(rho), np.mean(IG), np.std(IG), np.mean(ICI), np.std(ICI), np.mean(ISI), np.std(ISI)
# Perform infomax parameter sweep using multiprocessing:
with Pool(25) as p:
	results = np.asarray(p.starmap(info_block_sim, iterable))
# Format and save parameter sweep data as csv:
data_fields = ['epsilon', 'q', 'gamma', 'rho_mean', 'rho_std', 'IG_mean', 'IG_std', 'ICI_mean', 'ICI_std', 'ISI_mean', 'ISI_std']
data_rows = []
for i in range(len(index_dict)):
	data_rows.append([epsilon[index_dict[i][0]], q[index_dict[i][1]], gamma[index_dict[i][2]], results[i, 0], results[i, 1], results[i, 2], results[i, 3], results[i, 4], results[i, 5], results[i, 6], results[i, 7]])	
with open('im_high_h.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(data_fields)
    csvwriter.writerows(data_rows)

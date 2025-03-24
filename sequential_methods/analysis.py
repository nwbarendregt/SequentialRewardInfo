"""
analysis.py
Performs analysis on results of experimental simulation.
"""
import numpy as np
"""
accuracy():
Computes the agent's accuracy of commitments.
"""
def accuracy(results):
    return (np.sum(results.choice == results.correct_state)
            /len(results.correct_state))
"""
reward_rate():
Computes the agent's reward rate for the given experimental realization.
"""
def reward_rate(results):
    return np.sum(results.reward)/results.metadata.N
"""
sequence_lengths():
Computes the average commit burst and sample burst lengths.
"""
def sequence_lengths(results):
    # Pre-allocate storage of all commit burst and sample burst lengths:
    commit_length = np.array([])
    sample_length = np.array([])
    commit = 0
    for i in range(len(results.action)):
        if len(results.action[i]) == 1: # If agent commits on next action
            commit += 1 # Increment current commit burst length
        else: # If agent samples on next action
            sample_length = np.append(sample_length, len(results.action[i])-1) # Store sample burst length
            commit_length = np.append(commit_length, commit+1) # Store previous commit burst length
            commit = 0 # State new sample burst
    if len(commit_length) == 0: # If commit burst length is never stored (i.e., agent always commits)
        sample_length = np.array([0])
        commit_length = np.array([commit])    
    return np.mean(commit_length), np.mean(sample_length)
"""
information_gain():
Computes the information gained on each action as the uncertainty reduction caused by the agent's action.
"""
def information_gain(results):
    # Pre-allocate storage of information gains:
    IG = np.array([])
    # Extract agent's LLR belief trajectory:
    y = results.belief[~np.isnan(results.belief)]
    # Extract agent's action sequence:
    act = ''.join(results.action)
    for i in range(len(y)-1):
        if act[i] == 's': # If agent samples on the next action
            IG = np.append(IG, state_entropy(y[i])-state_entropy(y[i+1]))
        else: # If agent commits on the next action
            yp = np.log(((1-results.metadata.epsilon)*np.exp(y[i])+results.metadata.epsilon)
                        /(results.metadata.epsilon*np.exp(y[i])+1-results.metadata.epsilon))
            yp = y[i]
            IG = np.append(IG, state_entropy(yp)-state_entropy(y[i+1]))
    return np.append(IG, 0)
"""
state_entropy():
Helper function that computes the entropy over the current environmental state based on the agent's current LLR belief.
"""
def state_entropy(y):
    # Correct for infinite LLR edge cases:
    if np.abs(y) == np.inf:
        entropy = 0
    else:
        entropy = np.log2(1+np.exp(-y))/(1+np.exp(-y))+np.log2(np.exp(y)+1)/(np.exp(y)+1)
    return entropy   

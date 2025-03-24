"""
rewardmax_simulate.py
Performs a single realization of the rewardmax model for a specified task.
Generates plot trajectories in both state likelihood space (Fig. 2A) and action space (Fig. 2B).
"""
import numpy as np
import sequential_methods as sm
import matplotlib.pyplot as plt
# Define task and model parameters:
N = 10 # Time step budget
epsilon = 0.25; q = 0.75 # Switching probability and reward reliability
h = 0.75 # Bernoulli parameter for environmental evidence
R = (100, -100) # Reward structure (reward, punishment)
c = (1, 1) # Action time step costs (commitment, sampling)
gamma = 1 # Future utility discounting
# Generate task design object:
task_design = sm.task.BernoulliInference(N, epsilon, q, R, c, h)
# Generate agent's rewardmax model object:
objective = sm.model.RewardMax(gamma)
# Combine task and model objects:
behavior = sm.experiment.Behavior(task_design, objective)
# Simulate agent's behavior in task realization:
results = sm.experiment.simulate(behavior)
# Plot agent's behavior
t = np.arange(N, 0, -1) # Time step mesh for plotting
"""
Fig. 2A: Agent's belief realization in state likelihood space. 
"""
# Construct colormap for optimal action region visualization:
color_map = {0: np.array([141, 160, 203]),
             1: np.array([252, 141, 98])}
color_data = np.ndarray(shape=(behavior.opt_action.shape[0], behavior.opt_action.shape[1], 3), dtype=int)
for i in range(behavior.opt_action.shape[0]):
    for j in range(behavior.opt_action.shape[1]):
        color_data[i][j] = color_map[behavior.opt_action[i][j]]
extent = [10+0.5, 1-0.5, 0-0.001, 1+0.001]
fig, ax = plt.subplots()
# Plot optimal action regions in state likelihood space:
im = ax.imshow(color_data, interpolation = 'nearest',
               aspect='auto', extent=extent)
# Plot agent's belief trajectory in state likelihood space:
ax.plot(t, 1/(1+np.exp(-results.belief)), color='black', linewidth=5)
# Display state likelihood value p=0.5 for reference:
ax.plot(t, 0.5*np.ones_like(t), color = 'black', linestyle = '--')
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('RewardMax State Likelihood Trajectory')
ax.set_xlabel('Time Steps Used')
ax.set_xlim((N, 1))
ax.set_xticks(np.linspace(N, 1, 5), labels=['0', '', '', '', str(N-1)])
ax.set_ylabel('State Likelihood')
ax.set_yticks(np.linspace(0, 1, 5))
"""
Fig. 2B: Agent's action time series realization.
"""
# Sort rewarded and punished commitments and times:
RT_rewarded = results.RT[results.reward == R[0]]
RT_punished = results.RT[results.reward == R[1]]
choice_rewarded = results.choice[results.reward == R[0]]
choice_punished = results.choice[results.reward == R[1]]
fig, ax = plt.subplots()
# Plot rewarded commitment stems:
if len(RT_rewarded) > 0:
    reward_stem = ax.stem(RT_rewarded, choice_rewarded, markerfmt='go', linefmt='green')
    reward_stem[0].set_markersize(20)
    reward_stem[1].set_linewidth(5)
# Plot punished commitment stems:
if len(RT_punished) > 0:
    punish_stem = ax.stem(RT_punished, choice_punished, markerfmt='rX', linefmt='red')
    punish_stem[0].set_markersize(20)
    punish_stem[1].set_linewidth(5)
ax.plot(t, np.zeros_like(t), color='black')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.set_title('RewardMax Action Time Series')
ax.set_xlabel('Time Step Number')
ax.set_xlim((N+0.5, 0.5))
ax.set_xticks(np.arange(N, 0, -1), labels=['1']+['']*(N-2)+[str(N)])
ax.set_ylim((-1.1, 1.1))
ax.set_yticks([])
plt.show()

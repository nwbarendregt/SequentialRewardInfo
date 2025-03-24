"""
figure_3_generate.py
Generates alignment results shown in Fig. 3.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
epsilon = np.linspace(0, 0.5) # Switching probability
q = np.linspace(0.5, 1) # Reward reliability
gamma = [0, 0.25, 0.5, 0.75, 1] # Future utility discounting
# Load alignment data:
data = []
with open('alignment_standard.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        data.append(row)
data = np.array(data, dtype=float)
# Format alignment data for plotting:
alignment = np.array([])
for i in range(len(data)):
    alignment = np.append(alignment, data[i][-1])
alignment = np.reshape(alignment, (len(epsilon), len(q), len(gamma)))
"""
Fig. 3: Rewardmax and Infomax action alignment.
"""
for i in [0, 2, 4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(alignment[:, :, i])), cmap='Greys', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.linspace(cbar.vmin, cbar.vmax, 5))
    ax.set_title('Action Alignment, '+'$\gamma = $'+str(gamma[i])) 
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
plt.show()

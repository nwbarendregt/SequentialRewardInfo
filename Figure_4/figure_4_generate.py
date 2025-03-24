"""
figure_4_generate.py
Generates parameter sweep results shown in Fig. 4.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
epsilon = np.linspace(0, 0.5) # Switching probability
q = np.linspace(0.5, 1) # Reward reliability
gamma = [0, 0.25, 0.5, 0.75, 1] # Future utility discounting
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_standard.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_rho_mean = np.array([])
rm_rho_std = np.array([])
for i in range(len(rm_data)):
    rm_rho_mean = np.append(rm_rho_mean, rm_data[i][3])
    rm_rho_std = np.append(rm_rho_std, rm_data[i][4])
rm_rho_mean = np.reshape(rm_rho_mean, (len(epsilon), len(q), len(gamma)))
rm_rho_std = np.reshape(rm_rho_std, (len(epsilon), len(q), len(gamma)))
# Load infomax parameter sweep data:
im_data = []
with open('im_standard.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_rho_mean = np.array([])
im_rho_std = np.array([])
for i in range(len(im_data)):
    im_rho_mean = np.append(im_rho_mean, im_data[i][3])
    im_rho_std = np.append(im_rho_std, im_data[i][4])
im_rho_mean = np.reshape(im_rho_mean, (len(epsilon), len(q), len(gamma)))
im_rho_std = np.reshape(im_rho_std, (len(epsilon), len(q), len(gamma)))
"""
Fig. 4A: Average reward rate comparison.
"""
for i in [0, 4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose((rm_rho_mean[:, :, i]-im_rho_mean[:, :, i]))), cmap='bwr', norm=colors.CenteredNorm(), extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.linspace(cbar.vmin, cbar.vmax, 5))
    ax.set_title('Average Reward Rate Comparison, '+'$\gamma = $'+str(gamma[i])) 
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
""""
Fig. 4C: Reward rate distribution comparison.
"""
for i in [0, 4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose((rm_rho_mean[:, :, i]/rm_rho_std[:, :, i])-(im_rho_mean[:, :, i]/im_rho_std[:, :, i]))), cmap='bwr', norm=colors.CenteredNorm(), extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.linspace(cbar.vmin, cbar.vmax, 5))
    ax.set_title('Reward Rate Distribution Comparison, '+'$\gamma = $'+str(gamma[i]))  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
plt.show()

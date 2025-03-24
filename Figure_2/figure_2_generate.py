"""
figure_2_generate.py
Generates parameter sweep results shown in Fig. 2.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
# Construct approximate phase transition boundary:
h = 0.75 # Bernoulli parameter for environmental evidence
epsilon = np.linspace(0, 0.5) # Switching probability
q = np.linspace(0.5, 1) # Reward reliability
E, Q = np.meshgrid(epsilon, q)
Z = ((1-E)*Q+E*(1-Q))/(E*Q+(1-E)*(1-Q)) # Approximate phase transition boundary (see Appendix section of manuscript)
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
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_rho_mean = np.append(rm_rho_mean, rm_data[i][3])
    rm_rho_std = np.append(rm_rho_std, rm_data[i][4])
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_rho_mean = np.reshape(rm_rho_mean, (len(epsilon), len(q), len(gamma)))
rm_rho_std = np.reshape(rm_rho_std, (len(epsilon), len(q), len(gamma)))
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
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
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_rho_mean = np.append(im_rho_mean, im_data[i][3])
    im_rho_std = np.append(im_rho_std, im_data[i][4])
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_rho_mean = np.reshape(im_rho_mean, (len(epsilon), len(q), len(gamma)))
im_rho_std = np.reshape(im_rho_std, (len(epsilon), len(q), len(gamma)))
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
"""
Fig. 2C: Rewardmax explore-exploit behavioral phase space.
"""
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    ax.contour(E, Q, Z, levels=[h/(1-h)], colors='k', linewidths=5, linestyles='dashed')
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title('RewardMax Explore-Exploit Phase Space')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Fig. 2D: Infomax explore-exploit behavioral phase space.
"""
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    ax.contour(E, Q, Z, levels=[h/(1-h)], colors='k', linewidths=5, linestyles='dashed')
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title('InfoMax Explore-Exploit Phase Space')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
plt.show()

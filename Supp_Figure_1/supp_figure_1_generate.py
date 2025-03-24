"""
supp_figure_1_generate.py
Generates parameter sweep results shown in Supp Fig. 1.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
epsilon = np.linspace(0, 0.5) # Switching probability
q = np.linspace(0.5, 1) # Reward reliability
gamma = [0, 0.25, 0.5, 0.75, 1]
"""
Supp Fig. 1Ai: High commitment cost Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_c21.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $(\tau_c, \tau_c)=(2,1)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Aii: High sample cost Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_c12.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $(\tau_c, \tau_c)=(1,2)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Bi: Low Bernoulli parameter Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_low_h.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $h=0.55$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Bii: High Bernoulli Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_high_h.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $h=0.95$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Ci: High reward Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_R_biased.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $(R_c, R_i)=(110,\text{-}100)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Cii: High punishment Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_P_biased.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $(R_c, R_i)=(100,\text{-}110)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Di: Low budget Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_N5.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $N=5$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Dii: High budget Rewardmax behavior.
"""
# Load rewardmax parameter sweep data:
rm_data = []
with open('rm_N25.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        rm_data.append(row)
rm_data = np.array(rm_data, dtype=float)
# Format rewardmax parameter sweep data for plotting:
rm_ICI_mean = np.array([])
rm_ICI_std = np.array([])
rm_ISI_mean = np.array([])
rm_ISI_std = np.array([])
for i in range(len(rm_data)):
    rm_ICI_mean = np.append(rm_ICI_mean, rm_data[i][7])
    rm_ICI_std = np.append(rm_ICI_std, rm_data[i][8])
    rm_ISI_mean = np.append(rm_ISI_mean, rm_data[i][9])
    rm_ISI_std = np.append(rm_ISI_std, rm_data[i][10])
rm_ICI_mean = np.reshape(rm_ICI_mean, (len(epsilon), len(q), len(gamma)))
rm_ICI_std = np.reshape(rm_ICI_std, (len(epsilon), len(q), len(gamma)))
rm_ISI_mean = np.reshape(rm_ISI_mean, (len(epsilon), len(q), len(gamma)))
rm_ISI_std = np.reshape(rm_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(rm_ISI_mean[:, :, i]/rm_ICI_mean[:, :, i]), axis=1), cmap='Reds', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'RewardMax Explore-Exploit Phase Space, $N=25$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
plt.show()

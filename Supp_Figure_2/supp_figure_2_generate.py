"""
supp_figure_2_generate.py
Generates parameter sweep results shown in Supp Fig. 2.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
epsilon = np.linspace(0, 0.5) # Switching probability
q = np.linspace(0.5, 1) # Reward reliability
gamma = [0, 0.25, 0.5, 0.75, 1]
"""
Supp Fig. 2Ai: High commitment cost Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_c21.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $(\tau_c, \tau_c)=(2,1)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Aii: High sample cost Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_c12.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $(\tau_c, \tau_c)=(1,2)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Bi: Low Bernoulli parameter Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_low_h.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $h=0.55$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Bii: High Bernoulli Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_high_h.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $h=0.95$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Ci: High reward Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_R_biased.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $(R_c, R_i)=(110,\text{-}100)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Cii: High punishment Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_P_biased.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $(R_c, R_i)=(100,\text{-}110)$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Di: Low budget Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_N5.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $N=5$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
"""
Supp Fig. 1Dii: High budget Infomax behavior.
"""
# Load infomax parameter sweep data:
im_data = []
with open('im_N25.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvfile)
    for row in csvreader:
        im_data.append(row)
im_data = np.array(im_data, dtype=float)
# Format infomax parameter sweep data for plotting:
im_ICI_mean = np.array([])
im_ICI_std = np.array([])
im_ISI_mean = np.array([])
im_ISI_std = np.array([])
for i in range(len(im_data)):
    im_ICI_mean = np.append(im_ICI_mean, im_data[i][7])
    im_ICI_std = np.append(im_ICI_std, im_data[i][8])
    im_ISI_mean = np.append(im_ISI_mean, im_data[i][9])
    im_ISI_std = np.append(im_ISI_std, im_data[i][10])
im_ICI_mean = np.reshape(im_ICI_mean, (len(epsilon), len(q), len(gamma)))
im_ICI_std = np.reshape(im_ICI_std, (len(epsilon), len(q), len(gamma)))
im_ISI_mean = np.reshape(im_ISI_mean, (len(epsilon), len(q), len(gamma)))
im_ISI_std = np.reshape(im_ISI_std, (len(epsilon), len(q), len(gamma)))
for i in [4]:
    fig, ax = plt.subplots()
    pc = ax.imshow(np.flip(np.transpose(im_ISI_mean[:, :, i]/im_ICI_mean[:, :, i]), axis=1), cmap='Blues', origin='lower', extent=(0.5, 0, 0.5, 1))
    cbar = fig.colorbar(pc)
    cbar.set_ticks(ticks=np.append(1, np.linspace(cbar.vmin, cbar.vmax, 5)))
    ax.set_title(r'Infomax Explore-Exploit Phase Space, $N=25$')  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim(0.5-epsilon[1]/2, epsilon[1]/2)
    ax.set_xticks(np.linspace(0.5-epsilon[1]/2, epsilon[1]/2, 5))
    ax.set_ylabel('$q$')
    ax.set_ylim(0.5+epsilon[1]/2, 1-epsilon[1]/2)
    ax.set_yticks(np.linspace(0.5+epsilon[1]/2, 1-epsilon[1]/2, 5))
plt.show()

import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.chdir(os.getcwd())
S=49
A=4
R1 = np.load("R1.npy")
R2 = np.load("R2.npy")
T = np.load("T.npy")

r=R1+R2
terminal_state = np.load("terminal.npy")
start_state = np.load("initial.npy")
gamma = 0.9

def compute_q_values(S, A, R, T, gamma):
    # Initialize Q-values to zeros
    Q_new = np.zeros((S, A))
    
    # Maximum number of iterations for value iteration
    max_iterations = 5000
    
    # Value iteration
    for _ in range(max_iterations):
        Q = Q_new.copy()
        for s in range(S):
            for a in range(A):
                q_sa = 0
                for s_prime in range(S):
                    q_sa += T[s][a][s_prime] * (R[s][a][s_prime] + gamma * np.max(Q[s_prime]))
                Q_new[s][a] = q_sa
        if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
            print("Converged in", _ + 1, "iterations")
            break
        Q = Q_new
    
    return Q

Q1 = compute_q_values(S, A, R1, T, gamma)
a1 = np.argmax(Q1,axis=1)
Q2 = compute_q_values(S, A, R2, T, gamma)
a2 = np.argmax(Q2,axis=1)
Q3 = compute_q_values(S, A, R1+R2, T, gamma)
a3 = np.argmax(Q3,axis=1)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already loaded the required numpy arrays (R1, R2, T) and defined other parameters

# Function to convert state index to coordinates (x, y) assuming a 7x7 grid
def index_to_coords(state_idx, grid_size=7):
    return (state_idx // grid_size, state_idx % grid_size)

# Define actions: [Up, Down, Left, Right]
actions = ['Up', 'Down', 'Left', 'Right']
action_map = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}  # Action mapping to indices

# Initialize a grid to hold the policy (best action) for each state
policy_grid = np.full((7, 7), -1)  # Start with -1 (indicating no action)

# Fill the grid with the policies based on the computed a3 (combined policy)
for state_idx in range(S):
    x, y = index_to_coords(state_idx)  # Convert state index to (x, y) coordinates
    best_action = a2[state_idx]  # Best action based on combined reward (R1 + R2)
    policy_grid[x, y] = best_action

# Create a heatmap for the policy
plt.figure(figsize=(8, 6))
sns.heatmap(policy_grid, annot=True, cmap='coolwarm', cbar=False, 
            xticklabels=False, yticklabels=False, 
            center=1, vmin=-0.5, vmax=3.5, square=True)

# Map the actions to the corresponding labels
action_labels = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

# Customize plot to show the action labels
plt.title("Optimal Policy (R1 + R2 Combined)", fontsize=16)
plt.xlabel("X coordinate", fontsize=12)
plt.ylabel("Y coordinate", fontsize=12)

# Display action labels
for i in range(7):
    for j in range(7):
        action_idx = policy_grid[i, j]
        if action_idx != -1:
            plt.text(j + 0.5, i + 0.5, action_labels[int(action_idx)], ha='center', va='center', 
                     fontsize=12, color='black')

plt.show()

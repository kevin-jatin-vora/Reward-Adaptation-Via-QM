import numpy as np
import random
import os

class GridWorld:
    def __init__(self, grid_size, terminal_states, living_reward, max_bf=1, transition_prob=0.82):
        self.grid_size = grid_size
        self.num_states = grid_size[0] * grid_size[1]
        self.terminal_states = {self.to_1d(t[:2]): (t[2],t[3],t[4]) for t in terminal_states}  # {(x, y): reward}
        self.living_reward = living_reward
        self.max_bf = max_bf
        self.transition_prob = transition_prob
        
        self.actions = [
            (0, 0),         # stay
            (-1, 0), (-2, 0), # up 1, up 2
            (0, 1), (0, 2),   # right 1, right 2
            (0, -1), (0, -2)  # left 1, left 2
        ]
        self.state_values = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)  # Initially random policy (0 = up, 1 = down, 2 = left, 3 = right)
        
        # Transition matrix of shape (S, A, S) where S = num_states and A = 4 (number of actions)
        self.transition_matrix = np.zeros((self.num_states, len(self.actions), self.num_states))
        
        # Initialize the transition matrix
        self.init_transition_matrix()

    def to_1d(self, state):
        """ Convert (x, y) state to 1D index. """
        return state[0] * self.grid_size[1] + state[1]

    def to_2d(self, idx):
        """ Convert 1D index back to (x, y) state. """
        return (idx // self.grid_size[1], idx % self.grid_size[1])

    def boundary_check(self, state):
        """ Ensure the state does not go out of the grid boundaries. """
        x, y = state
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        return (x, y)

    def init_transition_matrix(self):
        """ Initialize the transition matrix based on max branching factor and transition probability. """
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                state = (x, y)
                state_idx = self.to_1d(state)
                
                if state_idx in self.terminal_states:
                    continue  # No transitions from terminal states
                
                # # Generate neighbors: number of neighbors can be between 0 and max_bf
                # num_neighbors = random.randint(0, self.max_bf)
                
                for action_idx, action in enumerate(self.actions):
                    next_state = self.boundary_check((x + action[0], y + action[1]))
                    next_state_idx = self.to_1d(next_state)
    
                    # Main transition to the next state (intended direction)
                    self.transition_matrix[state_idx][action_idx][next_state_idx] = self.transition_prob
    
                    # Generate neighbors: number of neighbors can be between 0 and max_bf
                    num_neighbors = random.randint(0, self.max_bf)
                    
                    # Allow repeated neighbors by selecting actions randomly and updating the transition matrix
                    temp_actions = self.actions.copy()
                    temp_actions.remove(action)
                    for _ in range(num_neighbors):
                        # if not temp_actions:
                        #     break  # If all actions are exhausted, stop
                        random_action = random.choice(temp_actions)
                        temp_actions.remove(random_action)  # Remove the action to avoid repeating it immediately
                        neighbor_state = self.boundary_check((x + random_action[0], y + random_action[1]))
                        neighbor_state_idx = self.to_1d(neighbor_state)
    
                        # Increment the probability for this neighbor, allowing repeats
                        self.transition_matrix[state_idx][action_idx][neighbor_state_idx] += (1 - self.transition_prob) / num_neighbors

                    # Normalize the probabilities to ensure they sum to 1
                    total_prob = np.sum(self.transition_matrix[state_idx][action_idx])
                
                    # Ensure that the transition probabilities sum to 1 by normalization
                    if total_prob > 0:
                        self.transition_matrix[state_idx][action_idx] /= total_prob

    def reward(self, state_idx):
        """ Return the reward for the given state (1D index). """
        return self.terminal_states.get(state_idx, (self.living_reward,self.living_reward, self.living_reward))

    def initialize_R1_and_R2(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                state = (x, y)
                state_idx = self.to_1d(state)
                for action_idx, action in enumerate(self.actions):
                    for x_dash in range(self.grid_size[0]):
                        for y_dash in range(self.grid_size[1]):
                            state_dash = (x_dash, y_dash)
                            state_dash_idx = self.to_1d(state_dash)
                            self.R1[state_idx,action_idx,state_dash_idx], self.R2[state_idx,action_idx,state_dash_idx], self.R3[state_idx,action_idx,state_dash_idx] = self.reward(state_dash_idx)
                            


# Grid size is 5x9, terminal states are [(0, 0, 1), (0, 8, 1), (4, 4, 1.2)]
grid_size = (7, 7)

STATE2WORLD = {
    0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6),
    7: (1, 0), 8: (1, 1), 9: (1, 2), 10: (1, 3), 11: (1, 4), 12: (1, 5), 13: (1, 6),
    14: (2, 0), 15: (2, 1), 16: (2, 2), 17: (2, 3), 18: (2, 4), 19: (2, 5), 20: (2, 6),
    21: (3, 0), 22: (3, 1), 23: (3, 2), 24: (3, 3), 25: (3, 4), 26: (3, 5), 27: (3, 6),
    28: (4, 0), 29: (4, 1), 30: (4, 2), 31: (4, 3), 32: (4, 4), 33: (4, 5), 34: (4, 6),
    35: (5, 0), 36: (5, 1), 37: (5, 2), 38: (5, 3), 39: (5, 4), 40: (5, 5), 41: (5, 6),
    42: (6, 0), 43: (6, 1), 44: (6, 2), 45: (6, 3), 46: (6, 4), 47: (6, 5), 48: (6, 6)
}
GOALS = [0, 7]  # state index of goals
WALLS = [  # state index of walls
    5, 6,
    14, 15,
    21, 22, 23, 24, 25,
    28, 29,
    46, 47, 48
]

terminal_states=[]
for s in GOALS+WALLS:
    if(s in GOALS):
        terminal_states.append(STATE2WORLD[s]+(0,0,2))
    if(s not in WALLS and s not in GOALS):
        terminal_states.append(STATE2WORLD[s]+(0,0.2,-0.3))
    if (s in WALLS):
        terminal_states.append(STATE2WORLD[s]+(0,-0.5,0.3))
    if(s==42):
        terminal_states.append(STATE2WORLD[s]+(3,0,-4))
        
        
# terminal_states = [(0, 0, 1, 0), (0, 8, 0, 1), (4, 4, 0.6, 0.6)]
living_reward = 0
max_bf = 0  # Example branching factor (can be 1 for deterministic)
transition_prob = 0.88

grid_world = GridWorld(grid_size, terminal_states, living_reward, max_bf, transition_prob)


root = input("Enter input path: ")
for i in range(3):
    os.mkdir(os.path.join(root,str(i)))
    for bf in range(0,7,2):
        grid_world = GridWorld(grid_size, terminal_states, living_reward, bf, transition_prob)
        np.save(os.path.join(os.path.join(root,str(i)),f'T_{bf+1}.npy'), grid_world.transition_matrix)

S=49
A=7
R1=np.zeros((S,A,S))
R2=np.zeros((S,A,S))
R3=np.zeros((S,A,S))
for s in range(S):
    for a in range(A):
        for sdash in range(S):
            if(sdash in GOALS):
                R3[s,a,sdash] = 2
            if(sdash not in WALLS and sdash not in GOALS):
                R2[s,a,sdash] = 0.2#0.9
                R3[s,a,sdash] = -0.3#-1
            if (sdash in WALLS):
                R2[s,a,sdash] = -0.5#-2.5
                R3[s,a,sdash] = 0.3#2.3
            if(sdash==42):
                R1[s,a,sdash] = 3
                R3[s,a,sdash] = -4
i=0                
np.save(os.path.join(os.path.join(root,str(i)),'R1.npy'), R1)
np.save(os.path.join(os.path.join(root,str(i)),'R2.npy'), R2)
np.save(os.path.join(os.path.join(root,str(i)),'R3.npy'), R3)
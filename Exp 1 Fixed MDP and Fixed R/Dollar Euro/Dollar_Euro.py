import numpy as np
import random
import os

class GridWorld:
    def __init__(self, grid_size, terminal_states, living_reward, max_bf=1, transition_prob=0.82):
        self.grid_size = grid_size
        self.num_states = grid_size[0] * grid_size[1]
        self.terminal_states = {self.to_1d(t[:2]): (t[2],t[3]) for t in terminal_states}  # {(x, y): reward}
        self.living_reward = living_reward
        self.max_bf = max_bf
        self.transition_prob = transition_prob
        
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self.state_values = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)  # Initially random policy (0 = up, 1 = down, 2 = left, 3 = right)
        
        # Transition matrix of shape (S, A, S) where S = num_states and A = 4 (number of actions)
        self.transition_matrix = np.zeros((self.num_states, len(self.actions), self.num_states))
        
        #initialize R1 and R2
        self.R1 = np.zeros((self.num_states, len(self.actions), self.num_states))
        self.R2 = np.zeros((self.num_states, len(self.actions), self.num_states))
        
        # Initialize the transition matrix
        self.init_transition_matrix()
        self.initialize_R1_and_R2()

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
        return self.terminal_states.get(state_idx, (self.living_reward,self.living_reward))

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
                            self.R1[state_idx,action_idx,state_dash_idx], self.R2[state_idx,action_idx,state_dash_idx] = self.reward(state_dash_idx)

# Example usage:

# Grid size is 5x9, terminal states are [(0, 0, 1), (0, 8, 1), (4, 4, 1.2)]
grid_size = (5, 9)
terminal_states = [(0, 0, 1, 0), (0, 8, 0, 1), (4, 4, 0.6, 0.6)]
living_reward = -0.0002
max_bf = 0  # Example branching factor (can be 1 for deterministic)
transition_prob = 0.856

grid_world = GridWorld(grid_size, terminal_states, living_reward, max_bf, transition_prob)
# grid_world.value_iteration()

# grid_world.print_values()
# grid_world.print_policy()

root = input("Enter input path: ")
for i in range(3):
    os.mkdir(os.path.join(root,str(i)))
    for bf in range(4):
        grid_world = GridWorld(grid_size, terminal_states, living_reward, bf, transition_prob)
        np.save(os.path.join(os.path.join(root,str(i)),f'T_{bf+1}.npy'), grid_world.transition_matrix)
    np.save(os.path.join(os.path.join(root,str(i)),'R1.npy'), grid_world.R1)
    np.save(os.path.join(os.path.join(root,str(i)),'R2.npy'), grid_world.R2)
    t_states=[]
    for t in terminal_states:
        t_states.append(grid_world.to_1d((t[0],t[1])))
    np.save(os.path.join(os.path.join(root,str(i)),"terminal.npy"),t_states)

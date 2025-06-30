import numpy as np
import random

class GridWorld:
    def __init__(self, grid_size=(7, 7)):
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right']
        self.transition_prob = 0.9  # Probability for intended action
        self.max_bf = 1  # Max branching factor (number of alternative actions)
        self.terminal_states = [self.to_1d((4, 2))]  # Target state is terminal
        self.start_state = [self.to_1d((1,5))]  # Initial state is start

        # Initialize transition matrix and reward matrices
        self.R1 = np.full((grid_size[0] * grid_size[1], len(self.actions), grid_size[0] * grid_size[1]), -.5)  # Default R1 values
        self.R2 = np.full((grid_size[0] * grid_size[1], len(self.actions), grid_size[0] * grid_size[1]), -.5)  # Default R2 values
        self.T = np.zeros((grid_size[0] * grid_size[1], len(self.actions), grid_size[0] * grid_size[1]))  # Transition matrix

        self.init_transition_matrix()
        self.set_rewards()

    def to_1d(self, state):
        """ Convert (x, y) to a 1D index. """
        return state[0] * self.grid_size[1] + state[1]

    def to_2d(self, index):
        """ Convert 1D index back to (x, y) state. """
        return (index // self.grid_size[1], index % self.grid_size[1])

    def boundary_check(self, state):
        """ Ensure the state does not go out of the grid boundaries. """
        x, y = state
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        return (x, y)

    def init_transition_matrix(self):
        """ Initialize the transition matrix with stochastic behavior. """
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                state = (x, y)
                state_idx = self.to_1d(state)
                
                if state_idx in self.terminal_states:
                    continue  # No transitions from terminal states

                for action_idx, action in enumerate(self.actions):
                    # Define movement for each action
                    if action == 'up':
                        next_state = (x - 1, y)
                    elif action == 'down':
                        next_state = (x + 1, y)
                    elif action == 'left':
                        next_state = (x, y - 1)
                    elif action == 'right':
                        next_state = (x, y + 1)
                    
                    # Ensure the next state is within boundaries
                    next_state = self.boundary_check(next_state)
                    next_state_idx = self.to_1d(next_state)
                    
                    # Main transition to the next state (intended direction)
                    self.T[state_idx, action_idx, next_state_idx] = self.transition_prob

                    # Handle other stochastic transitions
                    num_neighbors = random.randint(0, self.max_bf)
                    temp_actions = self.actions.copy()
                    temp_actions.remove(action)  # Remove the intended action to avoid repetition

                    for _ in range(num_neighbors):
                        if not temp_actions:
                            break  # If all actions are exhausted, stop
                        random_action = random.choice(temp_actions)
                        temp_actions.remove(random_action)
                        
                        # Calculate neighbor's next state based on random action
                        if random_action == 'up':
                            neighbor_state = (x - 1, y)
                        elif random_action == 'down':
                            neighbor_state = (x + 1, y)
                        elif random_action == 'left':
                            neighbor_state = (x, y - 1)
                        elif random_action == 'right':
                            neighbor_state = (x, y + 1)

                        # Ensure the neighbor state is within boundaries
                        neighbor_state = self.boundary_check(neighbor_state)
                        neighbor_state_idx = self.to_1d(neighbor_state)

                        # Increment the probability for this neighbor
                        self.T[state_idx, action_idx, neighbor_state_idx] += (1 - self.transition_prob) / (num_neighbors + 1)

                    # Normalize the transition probabilities for the current action
                    total_prob = np.sum(self.T[state_idx, action_idx])
                    if total_prob > 0:
                        self.T[state_idx, action_idx] /= total_prob

    def set_rewards(self):
        """ Set specific rewards for transitions based on provided paths. """
        # Transition 1 (path 1,5 → 0,5 → 0,4 → ... → 4,2)
        self.R1[self.to_1d((1, 5)), self.actions.index('up'), self.to_1d((0, 5))] = -0.995
        self.R2[self.to_1d((1, 5)), self.actions.index('up'), self.to_1d((0, 5))] = -0.005

        self.R1[self.to_1d((0, 5)), self.actions.index('left'), self.to_1d((0, 4))] = -0.995
        self.R2[self.to_1d((0, 5)), self.actions.index('left'), self.to_1d((0, 4))] = -0.005

        self.R1[self.to_1d((0, 4)), self.actions.index('left'), self.to_1d((0, 3))] = -0.995
        self.R2[self.to_1d((0, 4)), self.actions.index('left'), self.to_1d((0, 3))] = -0.005

        self.R1[self.to_1d((0, 3)), self.actions.index('left'), self.to_1d((0, 2))] = -0.995
        self.R2[self.to_1d((0, 3)), self.actions.index('left'), self.to_1d((0, 2))] = -0.005

        self.R1[self.to_1d((0, 2)), self.actions.index('left'), self.to_1d((0, 1))] = -0.995
        self.R2[self.to_1d((0, 2)), self.actions.index('left'), self.to_1d((0, 1))] = -0.005

        self.R1[self.to_1d((0, 1)), self.actions.index('left'), self.to_1d((0, 0))] = -0.995
        self.R2[self.to_1d((0, 1)), self.actions.index('left'), self.to_1d((0, 0))] = -0.005

        self.R1[self.to_1d((0, 0)), self.actions.index('down'), self.to_1d((1, 0))] = -0.995
        self.R2[self.to_1d((0, 0)), self.actions.index('down'), self.to_1d((1, 0))] = -0.005

        self.R1[self.to_1d((1, 0)), self.actions.index('down'), self.to_1d((2,0))] = -0.995
        self.R2[self.to_1d((1, 0)), self.actions.index('down'), self.to_1d((2,0))] = -0.005

        self.R1[self.to_1d((2,0)), self.actions.index('down'), self.to_1d((3, 0))] = -0.995
        self.R2[self.to_1d((2, 0)), self.actions.index('down'), self.to_1d((3, 0))] = -0.005

        self.R1[self.to_1d((3, 0)), self.actions.index('down'), self.to_1d((4, 0))] = -0.995
        self.R2[self.to_1d((3, 0)), self.actions.index('down'), self.to_1d((4, 0))] = -0.005

        self.R1[self.to_1d((4, 0)), self.actions.index('down'), self.to_1d((5, 0))] = -0.995
        self.R2[self.to_1d((4, 0)), self.actions.index('down'), self.to_1d((5, 0))] = -0.005

        self.R1[self.to_1d((5, 0)), self.actions.index('down'), self.to_1d((6, 0))] = -0.995
        self.R2[self.to_1d((5, 0)), self.actions.index('down'), self.to_1d((6, 0))] = -0.005

        self.R1[self.to_1d((6, 0)), self.actions.index('right'), self.to_1d((6, 1))] = -0.995
        self.R2[self.to_1d((6, 0)), self.actions.index('right'), self.to_1d((6, 1))] = -0.005
        
        self.R1[self.to_1d((6, 1)), self.actions.index('right'), self.to_1d((6, 2))] = -0.995
        self.R2[self.to_1d((6, 1)), self.actions.index('right'), self.to_1d((6, 2))] = -0.005
        
        self.R1[self.to_1d((6, 2)), self.actions.index('up'), self.to_1d((5, 2))] = -0.995
        self.R2[self.to_1d((6, 2)), self.actions.index('up'), self.to_1d((5, 2))] = -0.005
        
        # self.R1[self.to_1d((5, 2)), self.actions.index('up'), self.to_1d((4, 2))] = -0.995
        # self.R2[self.to_1d((5, 2)), self.actions.index('up'), self.to_1d((4, 2))] = -0.01
        
        self.R1[self.to_1d((1, 6)), self.actions.index('up'), self.to_1d((0, 6))] = -0.995
        self.R2[self.to_1d((1, 6)), self.actions.index('up'), self.to_1d((0, 6))] = -0.005
        self.R1[self.to_1d((0, 6)), self.actions.index('left'), self.to_1d((0, 5))] = -0.995
        self.R2[self.to_1d((0, 6)), self.actions.index('left'), self.to_1d((0, 5))] = -0.005
        
        self.R1[self.to_1d((2, 6)), self.actions.index('up'), self.to_1d((1, 6))] = -0.995
        self.R2[self.to_1d((2, 6)), self.actions.index('up'), self.to_1d((1, 6))] = -0.005
        
        self.R1[self.to_1d((3, 6)), self.actions.index('up'), self.to_1d((2, 6))] = -0.9995
        self.R2[self.to_1d((3, 6)), self.actions.index('up'), self.to_1d((2, 6))] = -0.0005
        
        self.R1[self.to_1d((4, 6)), self.actions.index('up'), self.to_1d((3, 6))] = -0.9995
        self.R2[self.to_1d((4, 6)), self.actions.index('up'), self.to_1d((3, 6))] = -0.0005
        
        self.R1[self.to_1d((4, 3)), self.actions.index('right'), self.to_1d((4, 4))] = -0.995
        self.R2[self.to_1d((4, 3)), self.actions.index('right'), self.to_1d((4, 4))] = -0.005
        self.R1[self.to_1d((4, 4)), self.actions.index('right'), self.to_1d((4, 5))] = -0.995
        self.R2[self.to_1d((4, 4)), self.actions.index('right'), self.to_1d((4, 5))] = -0.005
        self.R1[self.to_1d((4, 5)), self.actions.index('right'), self.to_1d((4, 6))] = -0.995
        self.R2[self.to_1d((4, 5)), self.actions.index('right'), self.to_1d((4, 6))] = -0.005
        
        self.R1[self.to_1d((4, 6)), self.actions.index('down'), self.to_1d((5, 6))] = -0.99
        self.R2[self.to_1d((4, 6)), self.actions.index('down'), self.to_1d((5, 6))] = -0.015
        self.R1[self.to_1d((5, 6)), self.actions.index('down'), self.to_1d((6, 6))] = -0.99
        self.R2[self.to_1d((5, 6)), self.actions.index('down'), self.to_1d((6, 6))] = -0.011
        self.R1[self.to_1d((6, 6)), self.actions.index('left'), self.to_1d((6, 5))] = -0.99
        self.R2[self.to_1d((6, 6)), self.actions.index('left'), self.to_1d((6, 5))] = -0.011
        self.R1[self.to_1d((6, 5)), self.actions.index('left'), self.to_1d((6, 4))] = -0.99
        self.R2[self.to_1d((6, 5)), self.actions.index('left'), self.to_1d((6, 4))] = -0.011
        self.R1[self.to_1d((6, 4)), self.actions.index('left'), self.to_1d((6, 3))] = -0.99
        self.R2[self.to_1d((6, 4)), self.actions.index('left'), self.to_1d((6, 3))] = -0.011
        self.R1[self.to_1d((6, 3)), self.actions.index('left'), self.to_1d((6, 2))] = -0.99
        self.R2[self.to_1d((6, 3)), self.actions.index('left'), self.to_1d((6, 2))] = -0.011
        
        # self.R1[self.to_1d((4, 5)), self.actions.index('down'), self.to_1d((5, 5))] = -0.9
        # self.R2[self.to_1d((4, 5)), self.actions.index('down'), self.to_1d((5, 5))] = -0.1
        # self.R1[self.to_1d((5, 5)), self.actions.index('down'), self.to_1d((6, 5))] = -0.9
        # self.R2[self.to_1d((5, 5)), self.actions.index('down'), self.to_1d((6, 5))] = -0.1
        
        ####################################################################################

        # Transition 2 (path 1,5 → 1,4 → 1,3 → ... → 4,2)
        self.R1[self.to_1d((1, 5)), self.actions.index('left'), self.to_1d((1, 4))] = -0.001
        self.R2[self.to_1d((1, 5)), self.actions.index('left'), self.to_1d((1, 4))] = -0.999
        
        self.R1[self.to_1d((1, 4)), self.actions.index('left'), self.to_1d((1, 3))] = -0.001
        self.R2[self.to_1d((1, 4)), self.actions.index('left'), self.to_1d((1, 3))] = -0.999

        self.R1[self.to_1d((1, 3)), self.actions.index('left'), self.to_1d((1, 2))] = -0.001
        self.R2[self.to_1d((1, 3)), self.actions.index('left'), self.to_1d((1, 2))] = -0.999

        self.R1[self.to_1d((1, 2)), self.actions.index('left'), self.to_1d((1, 1))] = -0.001
        self.R2[self.to_1d((1, 2)), self.actions.index('left'), self.to_1d((1, 1))] = -0.999

        self.R1[self.to_1d((1, 1)), self.actions.index('down'), self.to_1d((2, 1))] = -0.001
        self.R2[self.to_1d((1, 1)), self.actions.index('down'), self.to_1d((2, 1))] = -0.999

        self.R1[self.to_1d((2, 1)), self.actions.index('down'), self.to_1d((3, 1))] = -0.001
        self.R2[self.to_1d((2, 1)), self.actions.index('down'), self.to_1d((3, 1))] = -0.999

        self.R1[self.to_1d((3, 1)), self.actions.index('down'), self.to_1d((4, 1))] = -0.001
        self.R2[self.to_1d((3, 1)), self.actions.index('down'), self.to_1d((4, 1))] = -0.999

        # self.R1[self.to_1d((4, 1)), self.actions.index('right'), self.to_1d((4, 2))] = -0.05
        # self.R2[self.to_1d((4, 1)), self.actions.index('right'), self.to_1d((4, 2))] = -0.95
        self.R1[self.to_1d((2, 5)), self.actions.index('up'), self.to_1d((1, 5))] = -0.001
        self.R2[self.to_1d((2, 5)), self.actions.index('up'), self.to_1d((1, 5))] = -0.999
        
        self.R1[self.to_1d((3, 5)), self.actions.index('up'), self.to_1d((2, 5))] = -0.001
        self.R2[self.to_1d((3, 5)), self.actions.index('up'), self.to_1d((2, 5))] = -0.999
        
        self.R1[self.to_1d((4, 5)), self.actions.index('up'), self.to_1d((3, 5))] = -0.001
        self.R2[self.to_1d((4, 5)), self.actions.index('up'), self.to_1d((3, 5))] = -0.999
               
        self.R1[self.to_1d((2, 4)), self.actions.index('up'), self.to_1d((1, 4))] = -0.001
        self.R2[self.to_1d((2, 4)), self.actions.index('up'), self.to_1d((1, 4))] = -0.999
        
        self.R1[self.to_1d((3, 4)), self.actions.index('up'), self.to_1d((2, 4))] = -0.001
        self.R2[self.to_1d((3, 4)), self.actions.index('up'), self.to_1d((2, 4))] = -0.999
        
        self.R1[self.to_1d((4, 4)), self.actions.index('up'), self.to_1d((3, 4))] = -0.001
        self.R2[self.to_1d((4, 4)), self.actions.index('up'), self.to_1d((3, 4))] = -0.999
        
        # self.R1[self.to_1d((2, 3)), self.actions.index('up'), self.to_1d((1, 3))] = -0.05
        # self.R2[self.to_1d((2, 3)), self.actions.index('up'), self.to_1d((1, 3))] = -0.95
        
        # self.R1[self.to_1d((3, 3)), self.actions.index('up'), self.to_1d((2, 3))] = -0.05
        # self.R2[self.to_1d((3, 3)), self.actions.index('up'), self.to_1d((2, 3))] = -0.95
        
        # self.R1[self.to_1d((4, 3)), self.actions.index('up'), self.to_1d((3, 3))] = -0.05
        # self.R2[self.to_1d((4, 3)), self.actions.index('up'), self.to_1d((3, 3))] = -0.95
        
        
        self.R1[self.to_1d((1, 6)), self.actions.index('left'), self.to_1d((1, 5))] = -0.001
        self.R2[self.to_1d((1, 6)), self.actions.index('left'), self.to_1d((1, 5))] = -0.999
        
        self.R1[self.to_1d((2, 6)), self.actions.index('left'), self.to_1d((2, 5))] = -0.001
        self.R2[self.to_1d((2, 6)), self.actions.index('left'), self.to_1d((2, 5))] = -0.999
        
        self.R1[self.to_1d((3, 6)), self.actions.index('left'), self.to_1d((3, 5))] = -0.001
        self.R2[self.to_1d((3, 6)), self.actions.index('left'), self.to_1d((3, 5))] = -0.999
        ######################################################################################

        # Transition 3 (path 1,5 → 1,6 → 2,6 → 3,6 → 4,6 → 4,5 → 4,4 → 4,3 → 4,2)
        self.R1[self.to_1d((1, 5)), self.actions.index('right'), self.to_1d((1, 6))] = -0.3
        self.R2[self.to_1d((1, 5)), self.actions.index('right'), self.to_1d((1, 6))] = -0.3

        self.R1[self.to_1d((1, 6)), self.actions.index('down'), self.to_1d((2, 6))] = -0.3
        self.R2[self.to_1d((1, 6)), self.actions.index('down'), self.to_1d((2, 6))] = -0.3

        self.R1[self.to_1d((2, 6)), self.actions.index('down'), self.to_1d((3, 6))] = -0.3
        self.R2[self.to_1d((2, 6)), self.actions.index('down'), self.to_1d((3, 6))] = -0.3

        self.R1[self.to_1d((3, 6)), self.actions.index('down'), self.to_1d((4, 6))] = -0.3
        self.R2[self.to_1d((3, 6)), self.actions.index('down'), self.to_1d((4, 6))] = -0.3

        self.R1[self.to_1d((4, 6)), self.actions.index('left'), self.to_1d((4, 5))] = -0.3
        self.R2[self.to_1d((4, 6)), self.actions.index('left'), self.to_1d((4, 5))] = -0.3

        self.R1[self.to_1d((4, 5)), self.actions.index('left'), self.to_1d((4, 4))] = -0.3
        self.R2[self.to_1d((4, 5)), self.actions.index('left'), self.to_1d((4, 4))] = -0.3

        self.R1[self.to_1d((4, 4)), self.actions.index('left'), self.to_1d((4, 3))] = -0.3
        self.R2[self.to_1d((4, 4)), self.actions.index('left'), self.to_1d((4, 3))] = -0.3

        # self.R1[self.to_1d((4, 3)), self.actions.index('left'), self.to_1d((4, 2))] = -0.3
        # self.R2[self.to_1d((4, 3)), self.actions.index('left'), self.to_1d((4, 2))] = -0.3
        
        #terminal reward
        self.R1[self.to_1d((4, 3)), self.actions.index('left'), self.to_1d((4, 2))] = 0.5
        self.R2[self.to_1d((4, 3)), self.actions.index('left'), self.to_1d((4, 2))] = 0.5
        self.R1[self.to_1d((4, 1)), self.actions.index('right'), self.to_1d((4, 2))] = 0.5
        self.R2[self.to_1d((4, 1)), self.actions.index('right'), self.to_1d((4, 2))] = 0
        self.R1[self.to_1d((5, 2)), self.actions.index('up'), self.to_1d((4, 2))] = 0
        self.R2[self.to_1d((5, 2)), self.actions.index('up'), self.to_1d((4, 2))] = 0.5
        

    def print_matrices(self):
        """ Print the reward and transition matrices. """
        print("Transition Matrix (T) [State, Action, Next State]:")
        print(self.T.shape)
        print(self.T[0, 0])  # Example for state 0, action 'up' to next states

        print("Reward Matrix R1 (R1) [State, Action, Next State]:")
        print(self.R1.shape)
        print(self.R1[0, 0])  # Example for state 0, action 'up' to next state

        print("Reward Matrix R2 (R2) [State, Action, Next State]:")
        print(self.R2.shape)
        print(self.R2[0, 0])  # Example for state 0, action 'up' to next state


# Example usage
gw = GridWorld()

# Printing reward and transition matrices
# gw.print_matrices()


np.save("R1.npy", gw.R1)
np.save("R2.npy", gw.R2)
np.save("T.npy", gw.T)
np.save("terminal.npy", gw.terminal_states)
np.save("initial.npy", gw.start_state)
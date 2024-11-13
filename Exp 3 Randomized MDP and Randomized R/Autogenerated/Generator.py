import numpy as np
import os
from collections import defaultdict


class MDPGenerator:
    def __init__(self, n_actions, n_states):
        # self.max_bf = max_bf
        self.num_actions = n_actions
        self.num_states = n_states
        self.transitions = []
        self.rewards = np.ones((num_states, num_actions, num_states))*-0.001
        self.rewards2 = np.ones((num_states, num_actions, num_states))*-0.001
        self.terminal_state = list(np.random.randint(low=1,high=self.num_states, size=3))
        
    def generate_transitions_and_rewards(self, max_bf):
        self.transitions = []
        # Generate transition probabilities for each state-action pair
        for i in range(self.num_states):
            row = []
            for j in range(self.num_actions):
                action_probs = [0.0] * self.num_states  # Initialize all transition probabilities to 0
                
                # Randomly select `max_bf` states
                possible_states = np.random.choice(self.num_states, max_bf, replace=False)
                alphas =  [(len(possible_states)-1) * 5] + list(np.ones(len(possible_states)-1)) #list(np.ones(len(possible_states)))
                remaining_probs = np.random.dirichlet(alphas)
                # Create a list of tuples (state, remaining_prob)
                state_prob_pairs = list(zip(possible_states, remaining_probs))
              
                # Sort the list of tuples by remaining probability in descending order
                sorted_state_prob_pairs = sorted(state_prob_pairs, key=lambda x: x[1], reverse=True)
                
                # Extract the sorted lists of states and probabilities
                sorted_states = [state for state, _ in sorted_state_prob_pairs]
                sorted_probs = [prob for _, prob in sorted_state_prob_pairs]

                # other_states = possible_states[1:]    # The rest will get a portion of 0.3
                
                self.rewards[i,j,sorted_states[0]] = np.random.uniform(-5,-1)
                self.rewards2[i,j,sorted_states[0]] = np.random.uniform(1,5)
                
                self.rewards[i,j,sorted_states[1:]] = np.random.uniform(-2,-1)
                self.rewards2[i,j,sorted_states[1:]] = np.random.uniform(1,2)
                
                # alphas = [len(other_states) * 5] + list(np.ones(len(other_states)))
                
                # Distribute the probabilities among the states using Dirichlet distribution
                # remaining_probs = list(np.random.dirichlet(alphas))
                for idx, state in enumerate(sorted_states):
                    action_probs[state] = sorted_probs[idx]
                
                row.append(action_probs)
            self.transitions.append(row)
        self.transitions = np.array(self.transitions)

    def write_to_npy(self, dir_path):
        max_bf=16
        # Ensure the directory exists, create it if necessary
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save each array as an .npy file with the naming convention
        r1_path = os.path.join(dir_path, f'R1_{max_bf}.npy')
        r2_path = os.path.join(dir_path, f'R2_{max_bf}.npy')
        t_path = os.path.join(dir_path, f'T_{max_bf}.npy')
        terminal_path = os.path.join(dir_path, f'terminal_states_{max_bf}.npy')
        num_actions_path = os.path.join(dir_path, f'num_actions_{max_bf}.npy')
        num_states_path = os.path.join(dir_path, f'num_states_{max_bf}.npy')

        np.save(r1_path, self.rewards)    # Save R1
        np.save(r2_path, self.rewards2)   # Save R2
        np.save(t_path, self.transitions) # Save T
        np.save(terminal_path, self.terminal_state)  # Save terminal states
        np.save(num_actions_path, self.num_actions)  # Save num_actions
        np.save(num_states_path, self.num_states)    # Save num_states
    
            
        # print(f"Files saved in directory: {dir_path}")
        # print(f"R1 saved at: {r1_path}")
        # print(f"R2 saved at: {r2_path}")
        # print(f"Transitions saved at: {t_path}")
        # print(f"Terminal states saved at: {terminal_path}")


for i in range(3):
    # os.mkdir(str(i))
    num_actions = 8
    num_states = 50
    mdp_gen = MDPGenerator(num_actions, num_states)
    
    max_bf=num_actions
    folder = os.getcwd()
    os.chdir(os.getcwd())
    mdp_gen.generate_transitions_and_rewards(max_bf)
    mdp_gen.write_to_npy(os.path.join(folder,str(i)))  # Replace with actual directory path

    
    # Initialize tracker as defaultdict of lists
    tracker = defaultdict(list)
    
    for bf in range(1, 6, 2):
        # print(bf)
        # T = restrict_transition(mdp_gen.transitions, bf)
        T = mdp_gen.transitions
        
        n_values = np.random.randint(1, bf + 1, size=num_states)
        print(n_values.mean())
        
        mask = np.zeros_like(T)
        sorted_matrix = np.argsort(T, axis=2)
        for s in range(num_states):
            for a in range(num_actions):
                
                if(bf!=1):
                    # Get existing tracked states from tracker
                    existing_states = tracker[(s, a, bf-2)]
                    
                    # Determine how many more states to randomly select
                    num_existing = len(existing_states)
                    num_to_select = n_values[s] - num_existing
                    
                    if num_to_select > 0:
                        # Randomly select the remaining number of states from available states
                        available_states = np.setdiff1d(np.where(T[s, a] != 0)[0], existing_states)
                        new_states = np.random.choice(available_states, num_to_select, replace=False)
                        # Update the tracker with the newly selected states
                        # tracker[(s, a, bf)].extend(existing_states+new_states)
                        
                        # Combine existing and new states
                        selected_states = np.array(existing_states + new_states.tolist())
                        tracker[(s, a, bf)].extend(selected_states)
                    
                    elif num_to_select < 0:
                        # Too many existing states, so randomly select n_values[s] from existing states
                        selected_states = existing_states[:num_to_select]
                        tracker[(s, a, bf)].extend(existing_states)
                    
                    else:
                        # num_to_select == 0, use only the existing states
                        selected_states = np.array(existing_states)
                        tracker[(s, a, bf)].extend(existing_states)
                    
                    # Update the mask based on the selected states
                    mask[s, a, selected_states] = 1
                else:
                    mask[s, a, sorted_matrix[s, a, -1:]] = 1
                    tracker[(s, a, bf)].extend(list(sorted_matrix[s, a, -1:]))
        
        # Apply the mask
        restricted_matrix = T * mask
        
        # Normalize probabilities for the top n states for each action
        row_sums = restricted_matrix.sum(axis=2, keepdims=True)
        normalized_matrix = restricted_matrix / row_sums
        
        # Set terminal states' transitions to zero
        normalized_matrix[mdp_gen.terminal_state, :, :] = 0

        np.save(f"{i}//T_{bf}.npy",normalized_matrix)
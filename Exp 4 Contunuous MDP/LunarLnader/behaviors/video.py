import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time
import imageio
import gym
from gym.wrappers import RecordVideo
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

for suffix in ['b1', 'b2' ,'b1_mu','b2_mu']:
    # Load the DQN model
    state_size = 8
    action_size = 4
    seed = 0  # Random seed
    qnetwork_local = QNetwork(state_size, action_size, seed)
    qnetwork_local.load_state_dict(torch.load(f'checkpoint_{suffix}.pth'))
    frame_delay = 0.1
    import gym
    from gym.wrappers import RecordVideo
    # Create the MountainCar-v0 environment
    env = gym.make("LunarLander-v2",render_mode='rgb_array')
    env = RecordVideo(env, f'video_{suffix}',  episode_trigger = lambda episode_number: True)
    episodes = 10
    scores = []  # List to store scores for each episode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frames = []

    for i_episode in range(1, episodes+1):
        # print(i_episode)
        if(i_episode < 3):
            continue
        if(i_episode > 6):
            continue
        # Initialize a list to store frames for the current episode
        # episode_frames = []
        state = env.reset()[0]
        for t in range(2500):
            # Render the environment and append the frame
            # frame = env.render()
            
            # episode_frames.append(frame)

            st = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_values = qnetwork_local(st)
            action = np.argmax(action_values.cpu().data.numpy())
            next_state, reward, done, _,__ = env.step(action)
            
            state = next_state
            # print(t)
            if done:
                break

        # Extend the frames list with the episode_frames
        # frames.extend(episode_frames)

    # Save the recorded frames as an MP4 video
    # output_file = 'dqn.mp4'
    # imageio.mimsave(output_file, frames, fps=30)
    env.close()  # Close the environment after recording

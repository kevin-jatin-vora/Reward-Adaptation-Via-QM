
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time
import imageio

# Define QNetwork class
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

# Load the DQN model
state_size = 7
action_size = 3
seed = 0  # Random seed
qnetwork_local = QNetwork(state_size, action_size, seed)
qnetwork_local.load_state_dict(torch.load('checkpoint_RA.pth'))
frame_delay = 0.1
from ple import PLE
from ple.games.pong import Pong
game = Pong(MAX_SCORE=5)
env = PLE(game, fps=30, display_screen=True)
episodes=10
scores = []  # List to store scores for each episode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
frames = []

for i_episode in range(1, episodes+1):
    if(i_episode<3):
        continue
    if(i_episode>6):
        continue
    # Initialize a list to store frames for the current episode
    episode_frames = []
    env.reset_game()
    env.init()
    state = np.array(list(env.getGameState().values())).reshape(1, 7)
    score = 0
    for t in range(500):
    # while True:
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = qnetwork_local(st)
        action = np.argmax(action_values.cpu().data.numpy())
        # reward = env.act(action)
        # Execute the action based on the mapped action value
        if action == 0:  # Mapped action for moving paddle up
            reward = env.act(119)
        elif action == 1:  # Mapped action for moving paddle down
            reward = env.act(115)
        else:  # No action
            reward = env.act(None)
        next_state = np.array(list(env.getGameState().values()))
        # Capture the current frame and append it to the episode_frames list
        episode_frames.append(env.getScreenRGB())
        # time.sleep(frame_delay)
        done = env.game_over()
        state = next_state
        score += reward
        if done:
            break
    frames.extend(episode_frames)
# Save the recorded frames as an MP4 video
output_file = 'RA.mp4'
imageio.mimsave(output_file, frames, fps=30)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import math
from joblib import load
from concurrent.futures import ThreadPoolExecutor


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ClusterBoundsManager:
    def __init__(self, model_paths, action_size, device='cpu'):
        self.device = device
        self.kmeans = []
        self.bounds = []
        self.action_size = action_size

        for path in model_paths:
            behavior_models = []
            behavior_bounds = []

            for action in range(action_size):
                model_data = load(os.path.join(path, f'action_{action}.pkl'))
                kmeans = model_data['kmeans']
                stats = sorted(model_data['stats'], key=lambda x: x['cluster_label'])

                mins = torch.stack([torch.from_numpy(s['min']).float() for s in stats]).to(device)
                maxs = torch.stack([torch.from_numpy(s['max']).float() for s in stats]).to(device)

                behavior_models.append(kmeans)
                behavior_bounds.append({'min': mins, 'max': maxs})

            self.kmeans.append(behavior_models)
            self.bounds.append(behavior_bounds)

    def get_bounds(self, states, actions, behavior_idx=0):
        states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
        actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        n, state_dim = states.shape

        min_results = torch.zeros((n, state_dim), device=self.device)
        max_results = torch.zeros((n, state_dim), device=self.device)

        for action in range(self.action_size):
            mask = (actions_np == action)
            if mask.any():
                clusters = self.kmeans[behavior_idx][action].predict(states_np[mask])
                min_results[mask] = self.bounds[behavior_idx][action]['min'][clusters]
                max_results[mask] = self.bounds[behavior_idx][action]['max'][clusters]

        return min_results, max_results

    def get_combined_bounds(self, states, actions):
        all_mins = []
        all_maxs = []

        for i in range(len(self.kmeans)):
            min_b, max_b = self.get_bounds(states, actions, i)
            all_mins.append(min_b)
            all_maxs.append(max_b)

        combined_min = torch.stack(all_mins).min(dim=0)[0]
        combined_max = torch.stack(all_maxs).max(dim=0)[0]

        return combined_min, combined_max


class BoundsLearner:
    def __init__(self, state_size=8, action_size=4, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # State bounds (LunarLander-specific)
        self.state_low = torch.tensor([-1.5, -1.5, -0.5, -1.5, -5, -5, 0, 0], device=self.device)
        self.state_high = torch.tensor([1.5, 1.5, 1.5, 1.5, 5, 5, 1, 1], device=self.device)
        self.num_neighbors = 20

        # Cluster manager (paths to models saved by DQN_b1, DQN_b2)
        model_paths = [
            'regression idea for neighbours//clustering_models',        # from DQN_b1.py
            'regression idea for neighbours//clustering_models_2'       # from DQN_b2.py
        ]
        self.cluster_manager = ClusterBoundsManager(model_paths, action_size=self.action_size, device=self.device)

        self._initialize_networks()
        # Vectorized operations setup
        self._setup_vectorized_ops()

    def _setup_vectorized_ops(self):
        self.q_min = -1000.0
        self.q_max = 300.0

    def _initialize_networks(self):
        self.qnetwork_ub = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.qnetwork_lb = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.qnetwork_ub_target = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.qnetwork_lb_target = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)

        self.q1_ub = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.q2_ub = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.q1_lb = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.q2_lb = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)

        pretrained_paths = ['regression idea for neighbours//checkpoint_b1.pth', 'regression idea for neighbours//checkpoint_b2.pth', 'regression idea for neighbours//checkpoint_b1_mu.pth', 'regression idea for neighbours//checkpoint_b2_mu.pth']
        for net, path in zip([self.q1_ub, self.q2_ub, self.q1_lb, self.q2_lb], pretrained_paths):
            if os.path.exists(path):
                net.load_state_dict(torch.load(path, map_location=self.device))
                for param in net.parameters():
                    param.requires_grad = False

        self.optimizer_ub = optim.Adam(self.qnetwork_ub.parameters(), lr=0.0001)
        self.optimizer_lb = optim.Adam(self.qnetwork_lb.parameters(), lr=0.0001)
        self.soft_update(1.0)

    def initialize_networks(self, num_samples=800000000*2, batch_size=1024):
        """Initialize networks with reduced samples"""
        print("Initializing networks...")
        ub_losses, lb_losses = [], []
        
        for _ in range(num_samples // batch_size):
            states = torch.rand((batch_size, self.state_size), device=self.device)
            states = states * (self.state_high - self.state_low) + self.state_low
            
            with torch.no_grad():
                ub_targets = torch.clamp(self.q1_ub(states) + self.q2_ub(states), self.q_min, self.q_max)
                lb_targets = torch.clamp(self.q1_lb(states) + self.q2_lb(states), self.q_min, self.q_max)
                ub_targets = torch.max(ub_targets, lb_targets)
                lb_targets = torch.min(lb_targets, ub_targets)
            
            # UB update with Huber loss
            self.optimizer_ub.zero_grad()
            ub_outputs = self.qnetwork_ub(states)
            ub_loss = F.smooth_l1_loss(ub_outputs, ub_targets)
            ub_loss.backward()
            self.optimizer_ub.step()
            ub_losses.append(ub_loss.item())
            
            # LB update with Huber loss
            self.optimizer_lb.zero_grad()
            lb_outputs = self.qnetwork_lb(states)
            lb_loss = F.smooth_l1_loss(lb_outputs, lb_targets)
            lb_loss.backward()
            self.optimizer_lb.step()
            lb_losses.append(lb_loss.item())
        
        print(f"UB initialization loss: {np.mean(ub_losses):.4f}")
        print(f"LB initialization loss: {np.mean(lb_losses):.4f}")
        
        torch.save(self.qnetwork_ub.state_dict(), 'UB_init.pth')
        torch.save(self.qnetwork_lb.state_dict(), 'LB_init.pth')

    def get_neighbors(self, states, actions):
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(self.cluster_manager.get_bounds, states, actions, 0)
            f2 = executor.submit(self.cluster_manager.get_bounds, states, actions, 1)
            min1, max1 = f1.result()
            min2, max2 = f2.result()

        min_deltas, max_deltas = self.cluster_manager.get_combined_bounds(states, actions)
        rand_vals = torch.rand((len(states), self.num_neighbors, self.state_size), device=self.device)
        deltas = min_deltas.unsqueeze(1) + rand_vals * (max_deltas - min_deltas).unsqueeze(1)
        next_states = states.unsqueeze(1) + deltas
        return torch.clamp(next_states, self.state_low, self.state_high)
    
    def compute_reward(self, states, next_states):
        """
        Computes Gym-style reward using shaping difference.
        Inputs:
            states: Tensor of shape [B, 8] (s)
            next_states: Tensor of shape [B, 8] (s')
        Returns:
            reward: Tensor of shape [B, 1]
            done: Tensor of shape [B, 1]
        """
    
        def shaping_fn(s):
            x, y, vx, vy, angle, _, leg1, leg2 = torch.split(s, 1, dim=1)
            shaping = (
                -100 * torch.sqrt(x**2 + y**2)
                - 100 * torch.sqrt(vx**2 + vy**2)
                - 100 * torch.abs(angle)
                + 10 * (leg1 + leg2)
            )
            return shaping
    
        # Compute shaping at current and next state
        shaping_s = shaping_fn(states)
        shaping_s_prime = shaping_fn(next_states)
    
        # Base reward: shaping difference
        reward = shaping_s_prime - shaping_s
    
        # Terminal condition (done flag)
        x, y, vx, vy, angle, ang_vel, leg1, leg2 = torch.split(next_states, 1, dim=1)
        out_of_bounds = (torch.abs(x) > 1.5) | (torch.abs(y) > 1.5)
        below_surface = y <= 0
        done = out_of_bounds | below_surface
    
        # Final rewards: landing bonus or crash penalty
        safe_landing = (leg1 + leg2) >= 1.9
        crash_like = below_surface & (~safe_landing)
    
        reward += safe_landing.float() * 100.0
        reward -= crash_like.float() * 100.0
    
        return reward, done

    
    # def compute_reward(self, next_states):
    #     """Vectorized reward calculation"""
    #     x, y, vx, vy, angle, ang_vel, leg1, leg2 = torch.split(next_states, 1, dim=1)
        
    #     distance = torch.sqrt(x**2 + y**2)
    #     velocity = torch.sqrt(vx**2 + vy**2)
    #     angle_penalty = torch.abs(angle)
    #     leg_contact = leg1 + leg2
        
    #     landed = (y <= 0) & (leg_contact >= 1.9)
    #     crashed = (y <= 0) & ((velocity > 0.5) | (angle_penalty > 0.2))
        
    #     reward = -0.3 * distance - 0.03 * velocity - 0.1 * angle_penalty
    #     reward += landed.float() * (100 + 10 * (1 - velocity) + 10 * (1 - angle_penalty))
    #     reward -= crashed.float() * 100
    #     reward += (leg_contact > 0).float() * 10 * leg_contact
        
    #     done = (y <= 0) | (torch.abs(x) > 1.5) | (torch.abs(y) > 1.5)
        
    #     return reward, done

    def joint_update(self, batch_size=1024, gamma=0.99):
        """Vectorized joint update with stable training"""
    
        # Generate random states and actions
        states = torch.rand((batch_size, self.state_size), device=self.device)
        states = states * (self.state_high - self.state_low) + self.state_low
        actions = torch.randint(0, self.action_size, (batch_size,), device=self.device)
    
        # Generate neighbors and flatten for Q-value predictions
        next_states = self.get_neighbors(states, actions)  # [B, N, S]
        next_states_flat = next_states.view(-1, self.state_size)  # [B*N, S]
    
        # Compute rewards and dones for each neighbor
        # rewards, dones = self.compute_reward(next_states_flat)  # [B*N]
        states_expanded = states.unsqueeze(1).repeat(1, self.num_neighbors, 1).view(-1, self.state_size)
        rewards, dones = self.compute_reward(states_expanded, next_states_flat)  # both [1024, 8]

        rewards = rewards.view(batch_size, self.num_neighbors)
        dones = dones.view(batch_size, self.num_neighbors)
    
        # Current Q estimates
        current_q_ub = self.qnetwork_ub(states).gather(1, actions.unsqueeze(1))  # [B, 1]
        current_q_lb = self.qnetwork_lb(states).gather(1, actions.unsqueeze(1))  # [B, 1]
    
        with torch.no_grad():
            # Q-values for next states
            next_q_ub = self.qnetwork_ub_target(next_states_flat).view(batch_size, self.num_neighbors, -1)  # [B, N, A]
            next_q_lb = self.qnetwork_lb_target(next_states_flat).view(batch_size, self.num_neighbors, -1)  # [B, N, A]
    
            # Max Q-values across actions
            max_next_ub = next_q_ub.max(dim=2)[0]  # [B, N]
            max_next_lb = next_q_lb.max(dim=2)[0]  # [B, N]
    
            # Zero-out values where done
            max_next_ub[dones] = 0.0
            max_next_lb[dones] = 0.0
    
            # Compute UB and LB targets from reward + discounted future value
            targets_ub = (rewards + gamma * max_next_ub).max(dim=1)[0].unsqueeze(1)  # [B, 1]
            targets_lb = (rewards + gamma * max_next_lb).min(dim=1)[0].unsqueeze(1)  # [B, 1]
    
            # Enforce UB ≥ LB and clip to stability bounds
            targets_ub = torch.clamp(torch.max(targets_ub, targets_lb), self.q_min, self.q_max)
            targets_lb = torch.clamp(torch.min(targets_lb, targets_ub), self.q_min, self.q_max)
    
            # Additional constraint: shrink gap during training
            targets_ub = torch.min(targets_ub, current_q_ub)
            targets_lb = torch.max(targets_lb, current_q_lb)
    
        # Update UB network with Huber loss and gap penalty
        self.optimizer_ub.zero_grad()
        ub_loss = F.smooth_l1_loss(current_q_ub, targets_ub)
        penalty_ub = F.relu(current_q_lb.detach() - current_q_ub).pow(2).mean()
        (ub_loss + penalty_ub).backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_ub.parameters(), 1)
        self.optimizer_ub.step()
    
        # Update LB network with Huber loss and gap penalty
        self.optimizer_lb.zero_grad()
        lb_loss = F.smooth_l1_loss(current_q_lb, targets_lb)
        penalty_lb = F.relu(current_q_lb - current_q_ub.detach()).pow(2).mean()
        (lb_loss + penalty_lb).backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_lb.parameters(), 1)
        self.optimizer_lb.step()
    
        return ub_loss.item(), lb_loss.item()
    

    def soft_update(self, tau=0.001):
        """Soft update target networks"""
        for target_param, param in zip(self.qnetwork_ub_target.parameters(), 
                                     self.qnetwork_ub.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
        
        for target_param, param in zip(self.qnetwork_lb_target.parameters(), 
                                     self.qnetwork_lb.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
    
    def train(self, num_episodes=20000, update_every=4):
        """Optimized training loop"""
        s1=time.time()
        if not (os.path.exists('UB_init.pth') and os.path.exists('LB_init.pth')):
            self.initialize_networks()
        else:
            self.qnetwork_ub.load_state_dict(torch.load('UB_init.pth', map_location=self.device))
            self.qnetwork_lb.load_state_dict(torch.load('LB_init.pth', map_location=self.device))
        e1=time.time()
        file_path = "init_time.txt"        
        with open(file_path, 'w') as file:
            file.write(str(e1-s1))
            
        s2=time.time()
        for episode in range(num_episodes):
            ub_loss, lb_loss = self.joint_update()
            
            if episode % update_every == 0:
                self.soft_update()
            
            if episode % 100 == 0:
                print(f"Episode {episode}: UB Loss {ub_loss:.4f}, LB Loss {lb_loss:.4f}")
        
        torch.save(self.qnetwork_ub.state_dict(), 'UB_final_2new.pth')
        torch.save(self.qnetwork_lb.state_dict(), 'LB_final_2new.pth')
        e2=time.time()
        file_path = "iter_time.txt"        
        with open(file_path, 'w') as file:
            file.write(str(e2-s2))

if __name__ == "__main__":
    # print("sleeping. starting in 3 hrs from 7:25")
    # time.sleep(3*60*60)
    start_time = time.time()
    learner = BoundsLearner(state_size=8, action_size=4, seed=0)
    learner.train(num_episodes=10000)
    end_time = time.time()
    print(f"\nTotal training time: {end_time-start_time:.2f} seconds")
    file_path = "total_time.txt"        
    with open(file_path, 'w') as file:
        file.write(str(time.time()-start_time))
#gradient clipping changed from 0.5 to 1 and neighbours =4 instead of 2; name changed for pth file
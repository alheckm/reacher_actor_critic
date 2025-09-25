import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from sac_model import Actor, Critic

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
UPDATE_EVERY = 1        # how often to update the network (every UPDATE_EVERY timesteps)
NUM_UPDATES = 1         # how many times to update the network per learning step
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR_ACTOR = 3e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
LR_ALPHA = 3e-4         # learning rate of the temperature parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SACAgent():
    """Soft Actor-Critic Agent that interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, target_entropy=None):
        """Initialize an SAC Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            target_entropy (float): target entropy for automatic alpha tuning
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.t_step = 0
        
        # Target entropy for automatic temperature tuning
        if target_entropy is None:
            self.target_entropy = -action_size  # Heuristic value
        else:
            self.target_entropy = target_entropy

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Twin Critic Networks (local and target)
        self.critic1_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic2_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic1_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic2_target = Critic(state_size, action_size, random_seed).to(device)
        
        self.critic1_optimizer = optim.Adam(self.critic1_local.parameters(), lr=LR_CRITIC)
        self.critic2_optimizer = optim.Adam(self.critic2_local.parameters(), lr=LR_CRITIC)

        # Initialize targets to match local networks
        self.soft_update(self.critic1_local, self.critic1_target, 1.0)
        self.soft_update(self.critic2_local, self.critic2_target, 1.0)

        # Temperature parameter for entropy regularization
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward for each agent
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn every UPDATE_EVERY time steps, and do multiple updates
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, states, add_noise=True):
        """Returns actions for given states as per current policy for multiple agents."""
        return [self.act_single(state, add_noise) for state in states]

    def act_single(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            if add_noise:
                # Sample from stochastic policy
                action, _, _ = self.actor_local.sample(state)
            else:
                # Use deterministic mean for evaluation
                _, _, action = self.actor_local.sample(state)
        self.actor_local.train()
        return np.clip(action.cpu().data.numpy()[0], -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, π(next_state)) - α * log π(next_state))
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_state_actions, next_state_log_pis, _ = self.actor_local.sample(next_states)
            qf1_next_target = self.critic1_target(next_states, next_state_actions)
            qf2_next_target = self.critic2_target(next_states, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pis
            next_q_value = rewards + (gamma * (1 - dones) * min_qf_next_target)

        qf1 = self.critic1_local(states, actions)
        qf2 = self.critic2_local(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        qf1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        qf2_loss.backward()
        self.critic2_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        pi, log_pi, _ = self.actor_local.sample(states)

        qf1_pi = self.critic1_local(states, pi)
        qf2_pi = self.critic2_local(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------- update temperature ---------------------------- #
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1_local, self.critic1_target, TAU)
        self.soft_update(self.critic2_local, self.critic2_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def reset(self):
        """Reset the agent for a new episode."""
        pass  

    def save(self, checkpoint_path):
        """Save all network weights to a checkpoint file."""
        checkpoint = {
            'actor_local_state_dict': self.actor_local.state_dict(),
            'critic1_local_state_dict': self.critic1_local.state_dict(),
            'critic2_local_state_dict': self.critic2_local.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'target_entropy': self.target_entropy
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path):
        """Load network weights from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.actor_local.load_state_dict(checkpoint['actor_local_state_dict'])
        self.critic1_local.load_state_dict(checkpoint['critic1_local_state_dict'])
        self.critic2_local.load_state_dict(checkpoint['critic2_local_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.log_alpha.data.fill_(checkpoint['log_alpha'])
        self.alpha = self.log_alpha.exp()
        self.target_entropy = checkpoint['target_entropy']


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) 
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
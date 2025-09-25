import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """SAC Actor (Policy) Model - outputs mean and log_std for Gaussian policy."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        # Separate heads for mean and log_std
        self.mean_linear = nn.Linear(fc2_units, action_size)
        self.log_std_linear = nn.Linear(fc2_units, action_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mean_linear.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor network that maps states -> action distributions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std

    def sample(self, state):
        """Sample an action with reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Sample from the distribution
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Enforce action bound using tanh
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        
        return action, log_prob, mean


class Critic(nn.Module):
    """SAC Critic (Q-Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic network that maps (state, action) pairs -> Q-values."""
        x = torch.cat([state, action], 1)
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
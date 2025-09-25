# Reacher Continuous Control Project

This project trains agents to control robotic arms to reach target locations using Deep Deterministic Policy Gradient (DDPG) and Soft Actor-Critic (SAC) reinforcement learning algorithms.

## Project Details

### Environment Description

The Reacher environment is a Unity ML-Agents environment where a double-jointed arm moves to reach target locations in 3D space. The version used contains 20 identical agents, each with their own copy of the environment.

### State Space
- **Size**: 33-dimensional continuous state space
- **Contains**: Position, rotation, velocity, and angular velocities of the arm
- **Type**: Vector observation for each agent

### Action Space
- **Size**: 4-dimensional continuous action space
- **Actions**: Torque applicable to two joints
- **Range**: Each action is between -1 and 1

### Reward Structure
- **+0.1** reward for each step that the agent's hand is in the target location
- **Goal**: Maintain position at the target location for as many time steps as possible

### Solving Criteria
The environment is considered solved when the agents achieve an **average score of +30 over 100 consecutive episodes**, averaged over all 20 agents.

### Agents Implemented
This project includes two different continuous control algorithms:
1. **DDPG Agent** - Deep Deterministic Policy Gradient with experience replay, target networks, and Ornstein-Uhlenbeck noise
2. **SAC Agent** - Soft Actor-Critic with twin critics, entropy regularization, and automatic temperature tuning

## Getting Started

### Prerequisites
- Python 3.6 or higher
- PyTorch
- Unity ML-Agents toolkit
- NumPy, Matplotlib, and other scientific computing libraries

### Step 1: Set Up Python Environment

Create and activate a new conda environment with Python 3.6:

```bash
conda create --name drlnd python=3.6 -y
conda activate drlnd
```

### Step 2: Install PyTorch

Install a compatible PyTorch version (the original PyTorch 0.4.0 is no longer available):

```bash
pip install torch==1.7.1 torchvision==0.8.2
```

**Note**: This installs PyTorch 1.7.1 which is compatible with the Unity ML-Agents package, though newer than the originally specified version.

### Step 3: Install Unity ML-Agents

Clone the Udacity Continuous Control repository and install the Unity agents:

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install . --no-deps
pip install docopt pandas pytest pyyaml scipy matplotlib tqdm jupyter protobuf==3.5.2 grpcio==1.11.0
```

**Note**: We use `--no-deps` to avoid PyTorch version conflicts, then install the required dependencies separately. You can verify the installation by running:
```bash
python -c "from unityagents import UnityEnvironment; print('UnityEnvironment imported successfully')"
```

### Step 4: Download the Unity Environment

Download the appropriate Unity environment for your operating system:

**Version 2: Twenty (20) Agents** 
- **Linux**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- **Mac OSX**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- **Windows (32-bit)**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- **Windows (64-bit)**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

**For AWS users**: If training on AWS without a virtual screen, use [this headless version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip).

Extract the downloaded file and place it in the appropriate directory relative to your project.

### Step 5: Verify Installation

All required dependencies should now be installed. You can verify everything is working by testing the imports:
```bash
python -c "import torch; import numpy as np; from unityagents import UnityEnvironment; print('All imports successful! PyTorch version:', torch.__version__)"
```

## Instructions

### Training the Agents

To train both DDPG and SAC agents:

1. **Update the environment path** in `main.py`:
   ```python
   env = UnityEnvironment(file_name="path/to/your/Reacher_Linux/Reacher.x86_64")
   ```

2. **Activate the environment and run the training script**:
   ```bash
   conda activate drlnd
   python main.py
   ```

### Output Files

After training, you'll find:
- `ddpg_checkpoint.pth` - Trained DDPG model weights
- `sac_checkpoint.pth` - Trained SAC model weights
- `ddpg_scores.npy` - DDPG training scores array
- `sac_scores.npy` - SAC training scores array
- `ddpg_checkpoint_episodeX.pth` - Periodic DDPG checkpoints
- `sac_checkpoint_episodeX.pth` - Periodic SAC checkpoints

### Loading and Analyzing Results

To load and analyze training results run: 

```python
python analyze.py
```

which will generate a plot of the training scores:  `compare_training.png`
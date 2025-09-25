from unityagents import UnityEnvironment
from ddpg_agent import DDPGAgent
from sac_agent import SACAgent
import numpy as np
from tqdm import tqdm
from collections import deque
import torch


WINDOW_LENGTH = 100
TARGET_SCORE = 30.0

def train(env, agent, agent_name="Agent", n_episodes=300, max_t=1000, checkpoint_name="checkpoint.pth", save_every=25):
    """
    Params
    ======
        agent_name (str): Name of the agent for display
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        checkpoint_name (str): filename to save the trained weights
    """
    print(f"\n🌈 Training {agent_name}...")
    brain_name = env.brain_names[0]
    episode_scores = []               
    avg_score_window = deque(maxlen=WINDOW_LENGTH)  # last 100 scores
    
    # Create progress bar for the entire training
    pbar = tqdm(total=n_episodes, desc=f"Training {agent_name}", unit="episode")
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for _ in range(max_t):
            actions = agent.act(states)

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            
            agent.step(states, actions, rewards, next_states, dones)

            states = next_states                               # roll over states to next time step
            scores += env_info.rewards                         # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break

        episode_score = np.mean(scores) # take the average score across all agents
        episode_scores.append(episode_score)  
        avg_score_window.append(episode_score)
        avg_score = np.mean(avg_score_window)

        # Update progress bar with current stats
        pbar.set_postfix({
            'Avg Score': f'{avg_score:.2f}',
            'Current Score': f'{episode_score:.1f}'
        })
        pbar.update(1)
        
        # Print milestone every 100 episodes
        if i_episode % WINDOW_LENGTH == 0:
            pbar.write(f'Episode {i_episode}\tAverage Score: {avg_score:.2f}')
        
        # Save weights periodically
        if i_episode % save_every == 0:
            periodic_name = checkpoint_name.replace('.pth', f'_episode_{i_episode}.pth')
            agent.save(periodic_name)
            pbar.write(f'💾 Periodic save: {periodic_name}')
            
        if avg_score >= TARGET_SCORE:
            success_msg = f'\n🎉 {agent_name} solved environment in {i_episode} episodes! Average Score: {avg_score:.2f}'
            pbar.write(success_msg)
            pbar.close()
            agent.save(checkpoint_name)
            pbar.write(f'💾 Weights saved as {checkpoint_name}')
            break
    
    pbar.close()
    return episode_scores



if __name__ == "__main__":
    env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    sac_agent = SACAgent(state_size=state_size, action_size=action_size, random_seed=0)
    sac_scores = train(env, sac_agent, agent_name='SAC', checkpoint_name='sac_checkpoint.pth')
    np.save('sac_scores.npy', sac_scores)

    ddpg_agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=0)
    ddpg_scores = train(env, ddpg_agent, agent_name='DDPG', checkpoint_name='ddpg_checkpoint.pth', save_every=50)
    np.save('ddpg_scores.npy', ddpg_scores)

    env.close()

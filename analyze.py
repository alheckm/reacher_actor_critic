import numpy as np
import matplotlib.pyplot as plt
from main import TARGET_SCORE


def plot_scores(scores_sac, scores_ddpg, filename='training.png'):
    """Plot the scores and save to a file."""
    plt.figure(figsize=(10, 6))
    plt.plot(scores_sac,  color='blue', label='Scores SAC')
    plt.plot(scores_ddpg,  color='red', label='Scores DDPG')
    plt.axhline(y=TARGET_SCORE, color='green', linestyle='--', alpha=0.7, label=f'Target ({TARGET_SCORE})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Scores over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Plot saved: {filename}")
    
sac_scores = np.load('sac_scores.npy')
ddpg_scores = np.load('ddpg_scores.npy')
plot_scores(sac_scores, ddpg_scores, filename='compare_training.png')

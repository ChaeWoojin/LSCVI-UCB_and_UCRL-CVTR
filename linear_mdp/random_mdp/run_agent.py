import numpy as np
import matplotlib.pyplot as plt
from environments import LinearMDP
from agent import LSCVI_UCB_LinearMDP

def run_experiment():
    '''Run the γ-LSCVI-UCB algorithm on the Linear MDP environment.'''

    # Experiment setup
    T = 5000  # Episode length
    d = 8  # Dimensionality of the feature vector
    nState = 2  # Number of states
    nAction = 128  # Number of actions
    gamma = 1 - np.log(T) / np.sqrt(T)  # Discount factor
    lambda_reg = 1.0  # Regularization parameter
    delta = 0.2
    
    # Create the Linear MDP environment
    env_linear = LinearMDP(d, nState, nAction, gamma, T)
    init_s = env_linear.state
    phi = env_linear.phi
    H = env_linear.H  # Upper bound H is already computed in the environment
    beta = 20*H*d*np.sqrt(np.log(d*T/delta))  # Bonus coefficient

    # Initialize the agent (LSCVI-UCB)
    agent = LSCVI_UCB_LinearMDP(env_linear, init_s=init_s, gamma=gamma, phi=phi, lambda_reg=lambda_reg, H=H, beta=beta)
    total_reward_linear = agent.run()
    
    # Run the optimal policy
    total_reward_optimal = env_linear.run_optimal_policy()
    
    # Calculate regret
    regret = np.array(total_reward_optimal) - np.array(total_reward_linear)

    # Plot cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(total_reward_linear), label='LSCVI-UCB (Linear MDP)')
    plt.plot(np.cumsum(total_reward_optimal), label='Optimal Policy', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward vs Time')
    plt.legend()

    # Plot regret
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(regret), label='Regret')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title('Regret vs Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_experiment()
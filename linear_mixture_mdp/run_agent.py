import numpy as np
import matplotlib.pyplot as plt
from environments import LinearMixtureMDP
from agent_hack import UCRL_CVTR_hack
# from agent import UCRL_CVTR

def run_mixture_experiment():
    '''Run the UCRL-CVTR algorithm on the Linear Mixture MDP environment.'''

# Experiment setup
np.random.seed(0)
d = 8                               # dimensionality of the feature vector
nState = 2                          # number of states   
nAction = 128                       # number of actions
theta_star = np.random.rand(d)      # unknown probability kernel generating parameter
# print("theta_star: ", theta_star)
lambda_reg = 1                      # regularization parameter  
T = 10000                           # operating round
gamma = 0.9999                      # discounting factor
B = d

# Linear Mixture MDP experiment
env_mixture = LinearMixtureMDP(d, nState, nAction, theta_star, gamma, T)
init_s = env_mixture.state
phi = env_mixture.phi  # Retrieve phi from the environment before running the algorithm
H = env_mixture.compute_upper_bound_H()  # Compute the upper bound H from the environment

    # Initialize the agent (UCRL-CVTR)
    agent = UCRL_CVTR_hack(env_mixture, init_s=init_s, gamma=gamma, phi=phi, lambda_reg=lambda_reg, B=B, H=H)
    total_reward_mixture = agent.run()
    
    # Reset the environment and run the optimal policy
    env_mixture.reset()
    total_reward_optimal = env_mixture.run_optimal_policy_from_q_star()
    
    # Calculate regret
    regret = np.array(total_reward_optimal) - np.array(total_reward_mixture)

    # Plot cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(total_reward_mixture), label='UCRL-CVTR (Linear Mixture MDP)')
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
    run_mixture_experiment()

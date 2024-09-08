import numpy as np
import matplotlib.pyplot as plt
from environments import LinearMixtureMDP
from agent import UCRL_CVTR

def run_optimal_policy(env):
    '''Run the optimal policy based on q^* from the environment's compute_q_star_average_reward.'''
    total_reward_optimal = env.run_optimal_policy_from_q_star()
    return total_reward_optimal

# Experiment setup
np.random.seed(0)
d = 10                              # dimensionality of the feature vector
nState = 10                        # number of states   
nAction = 5                       # number of actions
theta_star = np.random.rand(d)      # unknown probability kernel generating parameter
lambda_reg = 1                      # regularization parameter  
T = 1000                            # operating round
gamma = 1-np.log(T)/np.sqrt(T)                        # discounting factor
B = d

# Linear Mixture MDP experiment
env_mixture = LinearMixtureMDP(d, nState, nAction, theta_star, gamma, T)
init_s = env_mixture.state
phi = env_mixture.phi  # Retrieve phi from the environment before running the algorithm
H = env_mixture.compute_upper_bound_H()  # Compute the upper bound H from the environment

# Run the UCRL-CVTR algorithm
agent_mixture = UCRL_CVTR(env_mixture, init_s, gamma, phi, lambda_reg, B, H)  # Pass phi and H as a parameter to the agent
total_reward_mixture = agent_mixture.run()

# Experiment setup remains the same
# Run the optimal policy with q^* using compute_q_star_average_reward
env_mixture.reset()
total_reward_optimal = run_optimal_policy(env_mixture)

# Calculate regret
regret = np.array(total_reward_optimal) - np.array(total_reward_mixture)

# Plot cumulative rewards and regret
plt.figure(figsize=(12, 6))

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

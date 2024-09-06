import numpy as np
import matplotlib.pyplot as plt
from environments import LinearMixtureMDP
from agent import UCRL_CVTR

def value_iteration(env, theta_star, gamma, tol=1e-6):
    '''Run value iteration to find the optimal policy for the true MDP.'''
    nState, nAction = env.nState, env.nAction
    V = np.zeros(nState)  # Initialize value function
    Q = np.zeros((nState, nAction))  # Initialize Q-values
    delta = float('inf')  # Initialize change in value function

    while delta > tol:  # Run value iteration until convergence
        delta = 0
        for s in range(nState):
            v = V[s]
            for a in range(nAction):
                # Compute transition probabilities using true theta_star
                transition_probs = np.dot(env.phi[s, a, :], theta_star)
                transition_probs /= np.sum(transition_probs)  # Normalize probabilities
                Q[s, a] = env.reward[s, a] + gamma * np.dot(transition_probs, V)

            V[s] = np.max(Q[s, :])  # Update value function with the best action's value
            delta = max(delta, abs(v - V[s]))  # Update delta to track convergence

    # Derive optimal policy from optimal value function
    optimal_policy = np.argmax(Q, axis=1)
    return optimal_policy


def run_optimal_policy(env, theta_star, T, gamma):
    '''Run the optimal policy using value iteration to find the optimal policy.'''
    # Use value iteration to find the optimal policy
    optimal_policy = value_iteration(env, theta_star, gamma)
    total_reward_optimal = []
    
    # Run the MDP using the optimal policy
    for t in range(T):
        s_t = env.state
        a_t = optimal_policy[s_t]  # Select the action from the optimal policy
        _, reward = env.step(s_t, a_t)  # Take the action and receive reward
        total_reward_optimal.append(reward)
    
    return total_reward_optimal


# Experiment setup
np.random.seed(0)
d = 10                              # dimensionality of the feature vector
nState = 500                        # number of states   
nAction = 200                       # number of actions
theta_star = np.random.rand(d)      # unknown probability kernel generating parameter
lambda_reg = 1                      # regularization parameter  
gamma = 0.99                        # discounting factor
T = 1000                            # operating round
B = d

# Linear Mixture MDP experiment
env_mixture = LinearMixtureMDP(d, nState, nAction, theta_star, gamma, T)
init_s = env_mixture.state
phi = env_mixture.phi  # Retrieve phi from the environment before running the algorithm
H = env_mixture.compute_upper_bound_H()  # Compute the upper bound H from the environment

# Run the UCRL-CVTR algorithm
agent_mixture = UCRL_CVTR(env_mixture, init_s, gamma, phi, lambda_reg, B, H)  # Pass phi and H as a parameter to the agent
total_reward_mixture = agent_mixture.run()

# Run the optimal policy with true theta_star using value iteration
env_mixture.reset()
total_reward_optimal = run_optimal_policy(env_mixture, theta_star, T, gamma)

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

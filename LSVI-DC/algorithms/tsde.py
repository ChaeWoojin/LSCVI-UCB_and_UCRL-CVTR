import numpy as np
from scipy.stats import dirichlet

class TSDE:
    def __init__(self, n_states, n_actions, prior_params):
        self.n_states = n_states
        self.n_actions = n_actions
        self.prior_params = prior_params
        
        self.t = 1
        self.episode = 0
        self.N = np.zeros((n_states, n_actions, n_states))
        self.episode_start = 0
        self.episode_length = 1
        
        self.theta = self.sample_model()
        self.policy = self.compute_optimal_policy(self.theta)
        
    def sample_model(self):
        theta = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                theta[s, a] = dirichlet.rvs(self.prior_params[s, a] + self.N[s, a])
        return theta
    
    def compute_optimal_policy(self, theta):
        # Implement value iteration or policy iteration here
        # This is a placeholder - you'd need to implement the actual MDP solver
        return np.random.randint(0, self.n_actions, size=self.n_states)
    
    def act(self, state):
        if self.t > self.episode_start + self.episode_length or \
           np.any(self.N[state] > 2 * self.N[state, :, :].sum(axis=1)):
            # Start new episode
            self.episode += 1
            self.episode_start = self.t
            self.episode_length = max(self.episode_length, self.t - self.episode_start)
            self.theta = self.sample_model()
            self.policy = self.compute_optimal_policy(self.theta)
        
        action = self.policy[state]
        return action
    
    def update(self, state, action, next_state):
        self.N[state, action, next_state] += 1
        self.t += 1

# Example usage:
n_states = 6
n_actions = 2
prior_params = np.ones((n_states, n_actions, n_states)) * 0.1  # Dirichlet prior

tsde = TSDE(n_states, n_actions, prior_params)

# Simulation loop
state = 0  # Start state
for _ in range(100000):  # Run for 100,000 steps
    action = tsde.act(state)
    next_state = np.random.choice(n_states)  # This should be your environment step
    tsde.update(state, action, next_state)
    state = next_state
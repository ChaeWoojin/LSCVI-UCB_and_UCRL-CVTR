import numpy as np
from tqdm import tqdm

class UCRL_CVTR_hack:
    def __init__(self, env, init_s, gamma, phi, lambda_reg, B, H):
        '''
        Initialize UCRL-CVTR agent.
        '''
        self.env = env
        self.state = init_s
        self.gamma = gamma
        self.phi = phi
        self.lambda_reg = lambda_reg
        self.H = H
        self.B = B
        self.T = env.T
        self.d = env.d
        self.delta = 0.01

        # Initialize Gram matrix and other parameters
        self.Sigma = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.N = self.T

    def run(self):
        '''Run the γ-UCRL-CVTR algorithm.'''
        total_reward = []
        with tqdm(total=self.T, desc="Total Timesteps", unit="timestep") as pbar:
            while self.env.timestep < self.T:
                t_k, theta_k, beta_k, Sigma_k, N_k = self.initialize_episode_params()
                V, Q = self.perform_value_iteration(N_k, theta_k)
                total_reward += self.execute_policy(Q, V, Sigma_k, t_k, N_k, pbar)
        return total_reward

    def initialize_episode_params(self):
        '''Initialize parameters for each episode.'''
        t_k = self.env.timestep
        theta_k = np.copy(self.theta_hat)
        beta_k = self.compute_confidence_bound(t_k)
        Sigma_k = np.copy(self.Sigma)
        N_k = self.T - t_k + 1
        return t_k, theta_k, beta_k, Sigma_k, N_k

    def compute_confidence_bound(self, t_k):
        '''Compute the confidence bound beta_k.'''
        return self.H * np.sqrt(self.d * np.log((self.lambda_reg + t_k * self.H) / (self.delta * self.lambda_reg))) + np.sqrt(self.lambda_reg) * self.B

    def perform_value_iteration(self, N_k, theta_k):
        '''Perform value iteration and return the updated Q and V values.'''
        V = np.zeros((N_k + 1, self.env.nState))
        Q = np.zeros((N_k + 1, self.env.nState, self.env.nAction))
        V[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
        Q[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)

        for n in tqdm(range(N_k), desc="Value Iteration", leave=False):
            V[n + 1, :], Q[n + 1, :, :] = self.value_iteration(V[n, :], Q[n, :, :], theta_k)

        return V, Q

    def execute_policy(self, Q, V, Sigma_k, t_k, N_k, pbar):
        '''Execute the learned policy and update rewards and parameters.'''
        total_reward = []
        while not self.terminate_episode(Sigma_k):
            s_t = self.env.state
            a_t = self.select_action_for_state(s_t, Q, N_k)

            s_next, reward = self.env.step(s_t, a_t)
            total_reward.append(reward)

            W_t = self.compute_W_t(V, a_t, s_t)
            phi_W = self.compute_phi_W(s_t, a_t, W_t)

            self.update_parameters(phi_W, W_t, s_next)
            pbar.update(1)
        return total_reward

    def terminate_episode(self, Sigma_k):
        '''Check if the episode should terminate based on Gram matrix determinant.'''
        return np.linalg.det(self.Sigma) > 2 * np.linalg.det(Sigma_k) or self.env.timestep >= self.env.T

    def value_iteration(self, V, Q, theta_k):
        '''Perform value iteration for a single step.'''
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                Q[s, a] = self.compute_Q(s, a, V, theta_k)
            V[s] = max(Q[s, :])
        return self.clip_value_function(V), Q

    def compute_Q(self, s, a, V, theta_k):
        '''Compute the Q-value for a given state-action pair using theta_k directly.'''
        probabilities = np.dot(self.phi[s, a, :], theta_k)
        if np.sum(probabilities) == 0:
            print(theta_k)
        else:
            probabilities /= np.sum(probabilities)  # Normalize
        return self.env.reward[s, a] + self.gamma * np.dot(probabilities, V)

    def select_action_for_state(self, s_t, Q, N_k):
        '''Select the action that maximizes the Q-value for the given state.'''
        xi_t = np.array([
            np.argmax([Q[i, s_t, a] for i in range(self.T - self.env.timestep + 1, N_k + 1)]) 
            + (self.T - self.env.timestep + 1) for a in range(self.env.nAction)
        ])
    
        # Return the action with the highest Q-value for the state s_t
        return np.argmax([Q[xi_t[a], s_t, a] for a in range(self.env.nAction)])

    def compute_W_t(self, V, a_t, s_t):
        '''Compute W_t for the current state-action pair.'''
        V_xi_1 = V[a_t - 1, :]
        return V_xi_1 - np.min(V_xi_1)

    def compute_phi_W(self, s_t, a_t, W_t):
        '''Compute the feature map \phi_{W_t}(s_t, a_t).'''
        return np.sum(self.phi[s_t, a_t, :] * W_t[:, np.newaxis], axis=0)

    def update_parameters(self, phi_W, W_t, s_next):
        '''Update the Gram matrix and b vector, then recalculate theta_hat.'''
        self.Sigma += np.outer(phi_W, phi_W)
        self.b += phi_W * W_t[s_next]
        self.update_theta()

    def update_theta(self):
        '''Update the estimate for θ_star (theta_hat) using ridge regression.'''
        self.theta_hat = np.linalg.solve(self.Sigma, self.b)

    def clip_value_function(self, V):
        '''Apply clipping to the value function to ensure span remains within bounds.'''
        min_V = np.min(V)
        return np.minimum(V, min_V + self.H)

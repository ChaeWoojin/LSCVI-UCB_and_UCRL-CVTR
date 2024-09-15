import numpy as np
from tqdm import tqdm

class LSVI_DC:
    def __init__(self, env, init_s, gamma, phi, lambda_reg, H, B, delta):
        '''
        Initialize LSVI_DC agent.
        '''
        self.env = env
        self.state = init_s
        self.gamma = gamma
        self.phi = phi
        self.lambda_reg = lambda_reg
        self.H = H
        self.B = B 
        self.delta = delta
        
        self.T = env.T
        self.d = env.d
        self.nState = env.nState
        self.nAction = env.nAction

        # self.beta = 16 * (self.H + 1) * env.d * np.sqrt(np.log(1 + ((self.d + 1)* self.T) / self.delta)) 
        self.beta = self.H * self.d
        self.Sigma = self.lambda_reg * np.eye(self.d + 1)  # Gram matrix
        self.history = []  # To store (s, a, s')


    def run(self):
        '''Run the LSVI_DC algorithm.'''
        total_reward = []
        prev_t_k = 0
        
        with tqdm(desc="Total Timesteps (LSVI_DC)", leave=False) as pbar:
            while self.env.timestep < self.T:
                t_k, Sigma_k, history_k, N_k = self.initialize_episode_params(prev_t_k)
                V_k, Q_k = self.perform_value_iteration(Sigma_k, history_k, N_k)
                total_reward += self.execute_policy(Q_k, V_k, Sigma_k, N_k, pbar)
                prev_t_k = t_k
            return total_reward
    
    def initialize_episode_params(self, prev_t_k):
        '''Initialize parameters for each episode.'''
        t_k = np.copy(self.env.timestep)
        Sigma_k = np.copy(self.Sigma)
        history_k = np.copy(self.history)
        
        prev_epi_size = t_k - prev_t_k
        if (prev_epi_size <= 10):
            N_k = 41
        else:
            N_k = prev_epi_size * 4

        if (N_k > self.T - self.env.timestep) : 
            N_k = self.T - self.env.timestep 
            
        return t_k, Sigma_k, history_k, N_k
        
    def perform_value_iteration(self, Sigma_k, history_k, N_k):
        '''Perform value iteration for current step.'''
        V_k = np.zeros((N_k + 1, self.env.nState))  # Initialize V
        Q_k = np.zeros((N_k + 1, self.env.nState, self.env.nAction))  # Initialize Q

        # Set the initial values of V and Q for step 0
        V_k[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
        Q_k[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)

        # Loop through each value iteration step
        for n in tqdm(range(N_k), desc="Value Iteration"):
            w_n_k = self.compute_weight_vector(V_k[n, :], Sigma_k, history_k)
            Q_k[n + 1, :, :], V_k[n + 1, :] = self.update_Q_V(w_n_k, V_k[n, :], Sigma_k)

        return V_k, Q_k

    def execute_policy(self, Q, V_k, Sigma_k, N_k, pbar):
        '''Execute policy to collect rewards and update history.'''
        total_reward = []
        Sigma_k = np.copy(Sigma_k)
        i = 0
        while np.linalg.det(self.Sigma) <= 2 * np.linalg.det(Sigma_k) and self.env.timestep < self.env.T:
            s_t = self.env.state
            a_t = self.select_action_for_state(Q, s_t, N_k, i)

            # Take the action, observe next state and reward
            s_next, reward = self.env.step(s_t, a_t)
            total_reward.append(reward)

            # Add transition to history and update Gram matrix
            self.history.append((s_t, a_t, s_next))
            self.update_gram_matrix(s_t, a_t)

            i += 1
            pbar.update(1)

        return total_reward

    def compute_weight_vector(self, V_k, Sigma_k, history):
        '''Compute the weight vector w_n^k for the current iteration.'''
        w_n_k = np.zeros(self.d + 1)
        for (s, a, s_next) in history:
            phi_sa = self.phi[s, a]
            delta_V = V_k[s_next] - np.min(V_k)
            w_n_k += np.dot(phi_sa, delta_V)

        return np.linalg.solve(Sigma_k, w_n_k)

    def update_Q_V(self, w_n_k, V_k, Sigma_k):
        '''Update Q and V using the weight vector and bonus term.'''
        Q_k = np.zeros((self.nState, self.nAction))
        Sigma_inv = np.linalg.inv(Sigma_k)  # Inverse of the Gram matrix

        for s in range(self.nState):
            for a in range(self.nAction):
                # Compute bonus term for uncertainty estimation
                phi_sa = self.phi[s, a, :]
                bonus = self.beta * np.sqrt(np.dot(np.dot(phi_sa.T, Sigma_inv), phi_sa))

                # Update Q function and apply value clipping
                Q_k[s, a] = min(
                    self.env.reward[s, a] + self.gamma * (np.dot(phi_sa, w_n_k) + np.min(V_k) + bonus),
                    1 / (1 - self.gamma)
                )

        V_k = np.max(Q_k, axis=1)
        V_k = np.minimum(V_k, np.min(V_k) + self.H)  # Apply clipping to V

        return Q_k, V_k

    def select_action_for_state(self, Q, s_t, N_k, i):
        '''Select the action that maximizes the Q-value for the given state.'''
        # Loop over all actions and find the best timestep that maximizes Q
        xi_t = np.array([
            np.argmax([Q[j, s_t, a] for j in range(i, N_k + 1)]) 
            for a in range(self.env.nAction)
        ])
        print([Q[xi_t[a], s_t, a] for a in range(self.env.nAction)])
        # Return the action with the highest Q-value for the state s_t
        return np.argmax([Q[xi_t[a], s_t, a] for a in range(self.env.nAction)])

    def update_gram_matrix(self, s, a):
        '''Update the Gram matrix Σ using the current state-action pair.'''
        phi_sa = self.phi[s, a, :]
        self.Sigma += np.outer(phi_sa, phi_sa)


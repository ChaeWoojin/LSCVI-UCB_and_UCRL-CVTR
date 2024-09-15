import numpy as np
from tqdm import tqdm

class LSCVI_UCB_LinearMDP:
    def __init__(self, env, init_s, gamma, phi, lambda_reg, H, beta):
        '''
        Initialize γ-LSCVI-UCB agent for linear MDP.
        Args:
            env - LinearMixtureMDP - the environment
            init_s - int - initial state
            gamma - float - discount factor
            phi - np.array - feature map provided by the environment
            lambda_reg - float - regularization parameter for the gram matrices
            H - float - upper bound for the value function (2 * sp(v^*))
            beta - float - bonus coefficient
        '''
        self.env = env
        self.state = init_s
        self.gamma = gamma
        self.phi = phi
        self.lambda_reg = lambda_reg
        self.H = H
        self.beta = beta

        self.T = env.T
        self.d = env.d
        self.nState = env.nState
        self.nAction = env.nAction

        # Initialize Gram matrix and history
        self.Sigma = self.lambda_reg * np.eye(self.d)  # Gram matrix
        self.history = []  # To store (s, a, s')

    def run(self):
        '''Run the γ-LSCVI-UCB algorithm.'''
        total_reward = []
        while self.env.timestep < self.T:
            V_k, Q_k, Sigma_k = self.perform_value_iteration()
            total_reward += self.execute_policy(Q_k, V_k, Sigma_k)

        return total_reward

    def perform_value_iteration(self):
        '''Perform value iteration for current step.'''
        N_k = self.T - self.env.timestep + 1  # Number of rounds remaining
        V_k = np.zeros((N_k + 1, self.env.nState))  # Initialize V
        Q_k = np.zeros((N_k + 1, self.env.nState, self.env.nAction))  # Initialize Q

        # Set the initial values of V and Q for step 0
        V_k[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
        Q_k[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)

        Sigma_k = np.copy(self.Sigma)
        history_k = np.copy(self.history)

        # Loop through each value iteration step
        for n in tqdm(range(N_k), desc="Value Iteration"):
            w_n_k = self.compute_weight_vector(V_k[n, :], Sigma_k, history_k)
            Q_k[n + 1, :, :], V_k[n + 1, :] = self.update_Q_V(w_n_k, V_k[n, :], Sigma_k)

        return V_k, Q_k, Sigma_k

    def execute_policy(self, Q_k, V_k, Sigma_k):
        '''Execute policy to collect rewards and update history.'''
        total_reward = []
        while np.linalg.det(self.Sigma) <= 2 * np.linalg.det(Sigma_k) and self.env.timestep < self.env.T:
            s_t = self.env.state
            a_t = self.select_best_action(Q_k, s_t)

            # Take the action, observe next state and reward
            s_next, reward = self.env.step(s_t, a_t)
            total_reward.append(reward)

            # Add transition to history and update Gram matrix
            self.history.append((s_t, a_t, s_next))
            self.update_gram_matrix(s_t, a_t)

            # Move to the next state
            self.state = s_next

        return total_reward

    def compute_weight_vector(self, V_k, Sigma_k, history):
        '''Compute the weight vector w_n^k for the current iteration.'''
        w_n_k = np.zeros(self.d)
        for (s, a, s_next) in history:
            phi_sa = self.phi[s, a, :]
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

    def select_best_action(self, Q_k, s_t):
        '''Select the action that maximizes Q-value for the current state.'''
        xi_t = np.zeros(self.env.nAction, dtype=int)
        for a in range(self.env.nAction):
            Q_values = [Q_k[i, s_t, a] for i in range(self.env.T - self.env.timestep + 1)]
            xi_t[a] = np.argmax(Q_values) + (self.env.T - self.env.timestep + 1)

        return np.argmax([Q_k[xi_t[a], s_t, a] for a in range(self.env.nAction)])

    def update_gram_matrix(self, s, a):
        '''Update the Gram matrix Σ using the current state-action pair.'''
        phi_sa = self.phi[s, a, :]
        self.Sigma += np.outer(phi_sa, phi_sa)

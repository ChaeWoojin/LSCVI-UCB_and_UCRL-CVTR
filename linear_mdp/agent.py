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

        # Initialize Gram matrix and historical set
        self.Sigma = self.lambda_reg * np.eye(self.d)  # Gram matrix
        self.history = []  # To store (s, a, s')

    def run(self):
        '''Run the γ-LSCVI-UCB algorithm.'''
        total_reward = []
        H = self.H

        while self.env.timestep < self.T:
            t_k = self.env.timestep
            Sigma_k = np.copy(self.Sigma)
            history_k = np.copy(self.history)
            N_k = self.T - t_k + 1  # Iteration rounds

            # Initialize V_k and Q_k values for value iteration
            V_k = np.zeros((N_k + 1, self.env.nState))
            Q_k = np.zeros((N_k + 1, self.env.nState, self.env.nAction))
            V_k[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
            Q_k[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)
            
            # Main loop for value iteration
            for n in tqdm(range(N_k), desc="Value Iteration"):
                # Update w_n^k for current n
                w_n_k = self.compute_weight_vector(V_k[n, :], Sigma_k, history_k)

                # Compute the updated Q and V values
                Q_k[n + 1, :, :], V_k[n + 1, :] = self.update_Q_V(w_n_k, V_k[n, :])

            # Policy execution
            while np.linalg.det(self.Sigma) <= 2 * np.linalg.det(Sigma_k) and self.env.timestep < self.env.T:
                s_t = self.env.state
                xi_t = np.zeros(self.env.nAction, dtype=int)  # Array to store \xi_t(a) for each action

                # For each action, find \xi_t(a)
                for a in range(self.env.nAction):
                    # Find \xi_t(a) as the timestep n that maximizes \widetilde{Q}(s_t, a)
                    Q_values = [Q_k[i, s_t, a] for i in range(self.T - self.env.timestep + 1, N_k + 1)]
                    xi_t[a] = np.argmax(Q_values) + (self.T - self.env.timestep + 1)  # Offset by starting index

                # Use \xi_t(a) to select the best action a_t
                a_t = self.select_action([Q_k[xi_t[a], s_t, a] for a in range(self.env.nAction)])

                # Take the action and receive the reward and next state
                s_next, reward = self.env.step(s_t, a_t)
                total_reward.append(reward)

                # Add the transition to history
                self.history.append((s_t, a_t, s_next))

                # Update Gram matrix with the new sample
                self.update_gram_matrix(self.state, a_t)

                # Move to the next state
                self.state = s_next

        return total_reward

    def compute_weight_vector(self, V_k, Sigma_k, history):
        '''Compute the weight vector w_n^k for current iteration.'''
        w_n_k = np.zeros(self.d)

        for (s, a, s_next) in history:
            phi_sa = self.phi[s, a, :]
            delta_V = V_k[s_next] - np.min(V_k)
            w_n_k += np.dot(phi_sa, delta_V)

        w_n_k = np.linalg.solve(Sigma_k, w_n_k)  # Apply the inverse of the Gram matrix
        return w_n_k

    def update_Q_V(self, w_n_k, V_k):
        '''Update Q and V using w_n_k and clipping.'''
        Q_k = np.zeros((self.nState, self.nAction))

        # Compute the inverse of the covariance matrix for the bonus term
        Sigma_inv = np.linalg.inv(self.Sigma)

        for s in range(self.nState):
            for a in range(self.nAction):
                # Calculate the bonus term: beta * ||phi(s, a)||_{Sigma^-1}
                phi_sa = self.phi[s, a, :]  # Feature vector for state-action pair (s, a)
                bonus = self.beta * np.sqrt(np.dot(np.dot(phi_sa.T, Sigma_inv), phi_sa))

                # Update Q function
                Q_k[s, a] = min(
                    self.env.reward[s, a] + self.gamma * (np.dot(self.phi[s, a, :], w_n_k) + np.min(V_k) + bonus),
                    1 / (1 - self.gamma)
                )

        # Update V function by taking the max over actions
        V_k = np.max(Q_k, axis=1)

        # Apply clipping to V
        V_k = np.minimum(V_k, np.min(V_k) + self.H)

        return Q_k, V_k


    def select_action(self, Q):
        '''Select the action that maximizes the Q-value.'''
        return np.argmax(Q)

    def update_gram_matrix(self, s, a):
        '''Update the Gram matrix Σ using the current state-action pair.'''
        phi_sa = self.phi[s, a, :]
        self.Sigma += np.outer(phi_sa, phi_sa)

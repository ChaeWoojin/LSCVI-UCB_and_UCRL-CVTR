import numpy as np
from tqdm import tqdm

class UCRL_CVTR:
    def __init__(self, env, init_s, gamma, phi, lambda_reg, B, H):
        '''
        Initialize UCRL-CVTR agent.
        Args:
            env - LinearMixtureMDP - the environment
            init_s - int - initial state
            gamma - float - discount factor
            phi - np.array - feature map provided by the environment
            lambda_reg - int - regularization parameter for the gram matrices
            H - float - upper bound for the value function
            beta_k - float - confidence bound scaling
            B - float - norm bound for theta
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
        self.Sigma = self.lambda_reg * np.eye(self.d)  # Gram matrix for variance
        self.b = np.zeros(self.d)                      # RHS for ridge regression
        self.theta_hat = np.zeros(self.d)              # Estimate for θ_star
        self.N = self.T                                # N as per Algorithm 2

    def run(self):
        '''Run the γ-UCRL-CVTR algorithm.'''
        total_reward = []
        N = self.N
        while self.env.timestep < self.T:
            # Set parameters
            t_k = self.env.timestep
            theta_k = np.copy(self.theta_hat)
            beta_k = self.H * np.sqrt(self.d * np.log((self.lambda_reg + t_k * self.H) / (self.delta * self.lambda_reg))) + np.sqrt(self.lambda_reg) * self.B
            Sigma_k = np.copy(self.Sigma)
            N_k = self.T - t_k + 1
            # print("\n t_k: {%d}\n" % t_k)

            # Initialize V and Q values for value iteration
            V = np.zeros((N_k + 1, self.env.nState))
            Q = np.zeros((N_k + 1, self.env.nState, self.env.nAction))
            V[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
            Q[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)

            # Perform value iteration with clipping
            for n in range(N_k):
                V[n + 1, :], Q[n + 1, :, :] = self.value_iteration(V[n, :], Q[n, :, :], theta_k, beta_k, Sigma_k)

            # Execution phase
            done = False
            while not done:
                s_t = self.env.state
                xi_t = np.zeros(self.env.nAction, dtype=int)  # Array to store \xi_t(a) for each action

                # For each action, find \xi_t(a)
                for a in range(self.env.nAction):
                    # Find \xi_t(a) as the timestep n that maximizes \widetilde{Q}(s_t, a)
                    # Iterate over the range [T - t + 1, N_k]
                    Q_values = [Q[i, s_t, a] for i in range(self.T - self.env.timestep + 1, N_k + 1)]
                    xi_t[a] = np.argmax(Q_values) + (self.T - self.env.timestep + 1)  # Offset by starting index

                # Use \xi_t(a) to select the best action a_t
                a_t = self.select_action([Q[xi_t[a], s_t, a] for a in range(self.env.nAction)])

                # print("t: %d, s_t: %d, a_t: %d, tau_t: %d" % (self.env.timestep, s_t, a_t, xi_t[a_t]))

                # Take the action and receive the reward and next state
                s_next, reward = self.env.step(s_t, a_t)
                total_reward.append(reward)

                # Compute W_t(s) = V_{(\xi_t - 1)}(s) - min V_{(\xi_t - 1)}
                V_xi_1 = V[xi_t[a_t] - 1, :]  # V at timestep \xi_t(a_t)
                W_t = V_xi_1 - np.min(V_xi_1)

                # Compute the feature map \phi_{W_t}(s_t, a_t)
                phi_W = np.sum(self.phi[s_t, a_t, :] * W_t[:, np.newaxis], axis=0)

                # Update Gram matrix and b vector (Ridge regression)
                self.Sigma += np.outer(phi_W, phi_W)  # Update \widehat{\Sigma}_{t+1}
                self.b += phi_W * W_t[s_next]  # Update \widehat{b}_{t+1}

                # Update theta_hat (parameter estimate)
                self.update_theta()

                # Check if episode is terminal
                # print("left: %f, right: %f" % (np.linalg.det(self.Sigma), np.linalg.det(Sigma_k)))
                if np.linalg.det(self.Sigma) > 2 * np.linalg.det(Sigma_k) or self.env.timestep >= self.env.T:
                    done = True

        return total_reward


    def value_iteration(self, V, Q, theta_k, beta_k, Sigma_k):
        '''Perform value iteration and clipping as per Algorithm 2.'''
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                # Maximize over the confidence set and compute Q
                Q[s, a] = self.compute_Q(s, a, V, theta_k, beta_k, Sigma_k)  # Pass beta_k and B here

            # Update value function
            V[s] = max(Q[s, :])

        # Clipping step to ensure span remains within bounds
        min_V = np.min(V)
        for s in range(self.env.nState):
            V[s] = min(V[s], min_V + self.H)

        return V, Q

    def solve_max_theta(self, s, a, V, theta_k, beta_k, Sigma_k):
        '''
        Solve the maximization problem: max_theta <theta, phi_V(s, a)> subject to
        theta in confidence set and ball constraint, ensuring theta induces valid probability kernel.
        
        Args:
            s - int - current state
            a - int - action
            V - np.array - current value function
            beta_k - float - confidence bound scaling
            Sigma_k - np.array - covariance matrix
        Returns:
            theta_star - np.array - optimal theta maximizing the inner product
        '''
        # Element-wise multiplication and summation
        phi_V = np.sum(self.phi[s, a, :] * V[:, np.newaxis], axis=0)

        # Compute the norm induced by Sigma_t (confidence set constraint)
        norm_sigma = np.sqrt(np.dot(np.dot(phi_V.T, np.linalg.inv(Sigma_k)), phi_V))

        # The optimal theta is in the direction of phi_V, scaled by the constraints
        theta_star = (beta_k / norm_sigma) * phi_V

        # Ensure that the probabilities sum to 1: Enforce sum of probabilities condition
        # Normalize theta_star to satisfy sum of probabilities = 1
        probabilities = np.dot(self.phi[s, a, :], theta_star)  # Compute the probabilities
        sum_probabilities = np.sum(probabilities)

        if sum_probabilities > 0:
            theta_star /= sum_probabilities  # Normalize theta_star to ensure probabilities sum to 1

        # Ensure non-negativity of probabilities: Set negative probabilities to 0
        probabilities = np.dot(self.phi[s, a, :], theta_star)  # Recompute probabilities
        probabilities[probabilities < 0] = 0  # Clip negative probabilities

        # Recalculate theta_star by solving a least squares problem with non-negative constraints
        # We solve argmin ||probabilities - phi[s, a, :] @ theta||_2, s.t. theta >= 0
        from scipy.optimize import lsq_linear
        res = lsq_linear(self.phi[s, a, :], probabilities, bounds=(0, np.inf))  # Ensure non-negativity
        theta_star = res.x

        return theta_star

    def compute_Q(self, s, a, V, theta_k, beta_k, Sigma_k):
        '''
        Compute the Q-value using the solution to the maximization problem.
        Args:
            s - int - state
            a - int - action
            V - np.array - value function
            theta_k - np.array - parameter estimate
            beta_k - float - confidence bound scaling
            Sigma_k - np.array - covariance matrix
        Returns:
            Q_value - float - the scalar Q-value for state-action pair (s, a)
        '''
        # Solve for the optimal theta in the confidence set
        theta_star = self.solve_max_theta(s, a, V, theta_k, beta_k, Sigma_k)

        # Compute the transition probabilities using theta_star
        probabilities = np.dot(self.phi[s, a, :], theta_star)
        # print(np.sum(probabilities))
        
        # Ensure that probabilities sum to 1
        probabilities /= np.sum(probabilities)

        # Compute the Q-value as the reward plus the expected value of the next states
        Q_value = self.env.reward[s, a] + self.gamma * np.dot(probabilities, V)

        return Q_value  # Ensure Q_value is a scalar


    def select_action(self, Q):
        '''Select the action that maximizes the Q-value.'''
        return np.argmax(Q)

    def update_theta(self):
        '''Update the estimate for θ_star (theta_hat) using ridge regression.'''
        self.theta_hat = np.linalg.solve(self.Sigma, self.b)

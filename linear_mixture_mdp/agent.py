import numpy as np
from tqdm import tqdm
import cvxpy as cp


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
        '''Run the γ-UCRL-CVTR algorithm with progress bar.'''
        total_reward = []
        N = self.N
        
        # Progress bar for episodes (i.e., timesteps)
        with tqdm(total=self.T, desc="Total Timesteps", unit="timestep") as pbar:
            while self.env.timestep < self.T:
                # Set parameters
                t_k = self.env.timestep
                theta_k = np.copy(self.theta_hat)
                beta_k = self.H * np.sqrt(self.d * np.log((self.lambda_reg + t_k * self.H) / (self.delta * self.lambda_reg))) + np.sqrt(self.lambda_reg) * self.B
                Sigma_k = np.copy(self.Sigma)
                N_k = self.T - t_k + 1

                # Initialize V and Q values for value iteration
                V = np.zeros((N_k + 1, self.env.nState))
                Q = np.zeros((N_k + 1, self.env.nState, self.env.nAction))
                V[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
                Q[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)

                # Perform value iteration with clipping
                for n in tqdm(range(N_k), desc="Value Iteration", leave=False):
                    V[n + 1, :], Q[n + 1, :, :] = self.value_iteration(V[n, :], Q[n, :, :], theta_k, beta_k, Sigma_k)

                # Execution phase
                done = False
                while not done:
                    s_t = self.env.state
                    xi_t = np.zeros(self.env.nAction, dtype=int)  # Array to store \xi_t(a) for each action

                    # For each action, find \xi_t(a)
                    for a in range(self.env.nAction):
                        # Find \xi_t(a) as the timestep n that maximizes \widetilde{Q}(s_t, a)
                        Q_values = [Q[i, s_t, a] for i in range(self.T - self.env.timestep + 1, N_k + 1)]
                        xi_t[a] = np.argmax(Q_values) + (self.T - self.env.timestep + 1)  # Offset by starting index

                    # Use \xi_t(a) to select the best action a_t
                    a_t = self.select_action([Q[xi_t[a], s_t, a] for a in range(self.env.nAction)])

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
                    if np.linalg.det(self.Sigma) > 2 * np.linalg.det(Sigma_k) or self.env.timestep >= self.env.T:
                        done = True

                    # Update the progress bar
                    pbar.update(1)

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
        theta_star = self.solve_max_theta_single_sa(s, a, V, theta_k, beta_k, Sigma_k)

        # Compute the transition probabilities using theta_star
        probabilities = np.dot(self.phi[s, a, :], theta_star)
        probabilities /= np.sum(probabilities)  # Ensure that probabilities sum to 1

        # Compute the Q-value as the reward plus the expected value of the next states
        Q_value = self.env.reward[s, a] + self.gamma * np.dot(probabilities, V)

        return Q_value

    def solve_max_theta_single_sa(self, s, a, V, theta_k, beta_k, Sigma_k):
        '''
        Solve the maximization problem for a single (s,a):
        max_theta <theta, phi_V> 
        subject to theta in confidence set and probability constraints for all (s,a).
        
        Args:
            s - int - state
            a - int - action
            V - np.array - current value function
            theta_k - np.array - empirical estimator for theta_star
            beta_k - float - confidence bound scaling
            Sigma_k - np.array - covariance matrix
        Returns:
            theta_star - np.array - optimal theta maximizing the inner product
        '''
        # Compute phi_V for the specific (s,a)
        phi_V = np.sum(self.phi[s, a, :] * V[:, np.newaxis], axis=0)  # shape (d,)
        
        # Define optimization variable for theta
        theta = cp.Variable(self.d)
        
        # Objective: maximize theta^T phi_V
        objective = cp.Maximize(phi_V @ theta)
        
        # Confidence set constraint: ||Sigma_k^{1/2}(theta - theta_k)||_2 <= beta_k
        constraints = [cp.quad_form(theta - theta_k, Sigma_k) <= beta_k**2]

        # Loop through all state-action pairs to ensure valid transition probabilities
        for s_prime in range(self.env.nState):
            for a_prime in range(self.env.nAction):
                # Get phi(s', a', ⋅)
                phi_sa_prime = self.phi[s_prime, a_prime, :]  # shape (nState, d)
                
                # Compute probabilities = phi(s', a', ⋅) @ theta
                probabilities_sa_prime = phi_sa_prime @ theta  # shape (nState,)

                # Probability constraint: Ensure non-negativity
                constraints.append(probabilities_sa_prime >= 0)

                # Probability constraint: Ensure the probabilities sum to 1
                constraints.append(cp.sum(probabilities_sa_prime) == 1)
        
        # Define the problem
        problem = cp.Problem(objective, constraints)
        
        # Solve the problem
        problem.solve(solver=cp.SCS)  # Choose appropriate solver
        
        # Check if a solution is found
        if theta.value is not None:
            theta_star = theta.value
        else:
            raise ValueError("Optimization did not converge")
        
        return theta_star

    def select_action(self, Q):
        '''Select the action that maximizes the Q-value.'''
        return np.argmax(Q)

    def update_theta(self):
        '''Update the estimate for θ_star (theta_hat) using ridge regression.'''
        self.theta_hat = np.linalg.solve(self.Sigma, self.b)

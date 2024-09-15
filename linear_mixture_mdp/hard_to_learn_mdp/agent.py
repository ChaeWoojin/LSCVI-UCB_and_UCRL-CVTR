import numpy as np
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB

class LSVI_DC:
    def __init__(self, env, init_s, gamma, phi, lambda_reg, B, delta):
        '''
        Initialize LSVI_DC agent.
        '''
        self.env = env
        self.state = init_s
        self.gamma = gamma
        self.phi = phi
        self.lambda_reg = lambda_reg
        self.H = env.D
        self.B = B
        self.T = env.T
        self.d = env.d
        self.delta = delta

        # Initialize Gram matrix and other parameters
        self.Sigma = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.N = self.T

        # Gurobi optimizer model
        self.gurobi_model = self.init_gurobi_model()

    def init_gurobi_model(self):
        '''Initialize the Gurobi model and set its parameters.'''
        model = gp.Model("solve_max_theta")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Define the optimization variable for theta
        theta = model.addMVar(self.d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")

        # Store the theta variable for future use
        self.theta_var = theta

        # Add probability constraints once
        for s_prime in range(self.env.nState):
            for a_prime in range(self.env.nAction):
                phi_sa_prime = self.phi[s_prime, a_prime, :]
                
                # Non-negativity constraints for probabilities
                model.addConstr(phi_sa_prime @ theta >= 0, name=f"probability_nonneg_{s_prime}_{a_prime}")
                
                # Probability sum constraint (probabilities should sum to 1)
                model.addConstr(gp.quicksum(phi_sa_prime @ theta) == 1, name=f"probability_sum_{s_prime}_{a_prime}")

        return model

    def run(self):
        '''Run the LSVI_DC algorithm.'''
        total_reward = []
        prev_t_k = 0  # Start time of the first episode (t_0 = 0)
        
        with open("estimates/LSVI_DC(LinMix).txt", "a", encoding='utf-8') as fw:
            fw.write("\n\n-----------------------------Start-----------------------------\n")
        
        with tqdm(desc="Total Timesteps (LSVI_DC)", leave=False) as pbar:
            while self.env.timestep < self.T:
                t_k, theta_k, beta_k, Sigma_k, N_k = self.initialize_episode_params(prev_t_k)
                with open("estimates/LSVI_DC(LinMix).txt", "a", encoding='utf-8') as fw:
                    theta_k_str = str(theta_k).replace('\n', ' ')
                    fw.write(' '.join([theta_k_str, str(self.env.timestep)]) + '\n')
                self.update_confidence_set_constraint(theta_k, beta_k, Sigma_k)
                V, Q = self.perform_value_iteration(N_k, theta_k, beta_k, Sigma_k)
                total_reward += self.execute_policy(Q, V, Sigma_k, N_k, pbar)

                prev_t_k = t_k

        return total_reward
  
    def update_confidence_set_constraint(self, theta_k, beta_k, Sigma_k):
        '''
        Update the quadratic confidence set constraint: ||Sigma_k^{1/2}(theta - theta_k)||_2 <= beta_k.
        This is done once per episode, so the constraints do not need to be modified repeatedly.
        '''
        # Confidence set constraint: ||Sigma_k^{1/2}(theta - theta_k)||_2 <= beta_k
        diff_theta = self.theta_var - theta_k

        # Use gp.QuadExpr to build the quadratic constraint ||Sigma_k^{1/2}(theta - theta_k)||_2^2
        Sigma_sqrt = np.linalg.cholesky(Sigma_k)
        quadratic_term = (Sigma_sqrt @ diff_theta) @ (Sigma_sqrt @ diff_theta)

        # Remove the previous confidence set constraint if it exists
        if 'confidence_set' in self.gurobi_model.getConstrs():
            self.gurobi_model.remove(self.gurobi_model.getConstrByName('confidence_set'))

        # Add the new quadratic constraint
        self.gurobi_model.addConstr(quadratic_term <= beta_k**2, name="confidence_set")

        # Warm-start optimization with the current theta_k
        self.theta_var.start = theta_k

    def initialize_episode_params(self, prev_t_k):
        '''Initialize parameters for each episode.'''
        t_k = self.env.timestep
        theta_k = np.copy(np.linalg.solve(self.Sigma, self.b))
        beta_k = self.compute_confidence_bound(t_k)
        Sigma_k = np.copy(self.Sigma)

        prev_epi_size = t_k - prev_t_k
        if (prev_epi_size <= 10):
            N_k = 41
        else:
            N_k = prev_epi_size * 4

        if (N_k > self.T - self.env.timestep) : 
            N_k = self.T - self.env.timestep 
            
        return t_k, theta_k, beta_k, Sigma_k, N_k

    def compute_confidence_bound(self, t_k):
        '''Compute the confidence bound beta_k.'''
        return self.H * np.sqrt(self.d * np.log((self.lambda_reg + t_k * self.H) / (self.delta * self.lambda_reg))) + np.sqrt(self.lambda_reg) * self.B

    def perform_value_iteration(self, N_k, theta_k, beta_k, Sigma_k):
        '''Perform value iteration and return the updated Q and V values.'''
        V = np.zeros((N_k + 1, self.env.nState))
        Q = np.zeros((N_k + 1, self.env.nState, self.env.nAction))
        V[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
        Q[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)

        for n in tqdm(range(N_k), desc="Value Iteration (LSVI-DC)", leave=False):
            V[n + 1, :], Q[n + 1, :, :] = self.value_iteration(V[n, :], Q[n, :, :], theta_k, beta_k, Sigma_k)

        return V, Q

    def execute_policy(self, Q, V, Sigma_k, N_k, pbar):
        '''Execute the learned policy and update rewards and parameters.'''
        total_reward = []
        Sigma_k = np.copy(Sigma_k)
        i = 0
        while not self.terminate_episode(Sigma_k):
            # print(i, np.linalg.det(self.Sigma), np.linalg.det(Sigma_k))
            s_t = self.env.state
            xi_t, a_t = self.select_action_for_state(Q, s_t, N_k, i)

            s_next, reward = self.env.step(s_t, a_t)
            # print(reward)
            total_reward.append(reward)

            W_t = self.compute_W_t(V, xi_t, a_t)
            phi_W = self.compute_phi_F(s_t, a_t, W_t)

            self.Sigma += np.outer(phi_W, phi_W)
            self.b += phi_W * W_t[s_next]
            i += 1
            pbar.update(1)
        return total_reward

    def terminate_episode(self, Sigma_k):
        '''Check if the episode should terminate based on Gram matrix determinant.'''
        return np.linalg.det(self.Sigma) > 2 * np.linalg.det(Sigma_k) or self.env.timestep >= self.env.T

    def value_iteration(self, V, Q, theta_k, beta_k, Sigma_k):
        '''Perform value iteration for a single step.'''
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                Q[s, a] = self.compute_Q(s, a, V, theta_k, beta_k, Sigma_k)
            V[s] = max(Q[s, :])
        return self.clip_value_function(V), Q

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
        Solve the maximization problem for a single (s,a) using Gurobi:
        max_theta <theta, phi_V> 
        subject to theta in confidence set and probability constraints for all (s,a).
        '''
        # Compute phi_V for the specific (s,a)
        phi_V = self.compute_phi_F(s, a, V)  # shape (d,)

        # Update the Gurobi model objective function
        self.gurobi_model.setObjective(phi_V @ self.theta_var, GRB.MAXIMIZE)

        # Optimize the model
        self.gurobi_model.optimize()

        # Check if a solution is found
        if self.gurobi_model.status == GRB.OPTIMAL:
            theta_star = self.theta_var.X  # Extract the optimal solution
        else:
            raise ValueError("Optimization did not converge")

        return theta_star

    def select_action_for_state(self, Q, s_t, N_k, i):
        '''Select the action that maximizes the Q-value for the given state.'''
        # Loop over all actions and find the best timestep that maximizes Q
        xi_t = np.array([
            np.argmax([Q[j, s_t, a] for j in range(i, N_k + 1)]) 
            for a in range(self.env.nAction)
        ])
    
        # Return the action with the highest Q-value for the state s_t
        return xi_t, np.argmax([Q[xi_t[a], s_t, a] for a in range(self.env.nAction)])

        # try:
        #     xi_t = np.array([
        #         np.argmax([Q[j, s_t, a] for j in range(i, N_k + 1)]) 
        #         for a in range(self.env.nAction)
        #     ])
        
        #     # Return the action with the highest Q-value for the state s_t
        #     return xi_t, np.argmax([Q[xi_t[a], s_t, a] for a in range(self.env.nAction)])
        # except:
        #     print("error: ", i, N_k)
        #     return 0

    def compute_W_t(self, V, xi_t, a_t):
        '''Compute W_t for the current state-action pair.'''
        V_xi_1 = V[xi_t[a_t] - 1, :]
        return V_xi_1 - np.min(V_xi_1)

    def compute_phi_F(self, s_t, a_t, F):
        '''Compute the feature map \phi_{W_t}(s_t, a_t).'''
        return np.sum(self.phi[s_t, a_t, :] * F[:, np.newaxis], axis=0)

    def clip_value_function(self, V):
        '''Apply clipping to the value function to ensure span remains within bounds.'''
        min_V = np.min(V)
        return np.minimum(V, min_V + self.H)

class UCRL2_VTR:
    def __init__(self, env, init_s, gamma, phi, lambda_reg, B, delta):
        '''
        Initialize UCRL-CVTR agent.
        '''
        self.env = env
        self.state = init_s
        self.gamma = gamma
        self.phi = phi
        self.lambda_reg = lambda_reg
        self.D = env.D
        self.B = B
        self.T = env.T
        self.d = env.d
        self.delta = delta

        # Initialize Gram matrix and other parameters
        self.Sigma = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.N = self.T

        # Gurobi optimizer model
        self.gurobi_model = self.init_gurobi_model()

    def init_gurobi_model(self):
        '''Initialize the Gurobi model and set its parameters.'''
        model = gp.Model("solve_max_theta")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Define the optimization variable for theta
        theta = model.addMVar(self.d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")

        # Store the theta variable for future use
        self.theta_var = theta

        # Add probability constraints once
        for s_prime in range(self.env.nState):
            for a_prime in range(self.env.nAction):
                phi_sa_prime = self.phi[s_prime, a_prime, :]
                
                # Non-negativity constraints for probabilities
                model.addConstr(phi_sa_prime @ theta >= 0, name=f"probability_nonneg_{s_prime}_{a_prime}")
                
                # Probability sum constraint (probabilities should sum to 1)
                model.addConstr(gp.quicksum(phi_sa_prime @ theta) == 1, name=f"probability_sum_{s_prime}_{a_prime}")

        return model
    
    def run(self):
        '''Run the γ-UCRL-CVTR algorithm.'''
        total_reward = []
        prev_t_k = 0  # Start time of the first episode (t_0 = 0)
        
        with open("estimates/UCRL2-VTR(LinMix).txt", "a", encoding='utf-8') as fw:
            fw.write("\n\n-----------------------------Start-----------------------------\n")

        with tqdm(desc="Total Timesteps (UCRL2_VTR)", leave=False) as pbar:
            while self.env.timestep < self.T:
                t_k, theta_k, beta_k, Sigma_k, N_k = self.initialize_episode_params(prev_t_k)
                self.update_confidence_set_constraint(theta_k, beta_k, Sigma_k) 
                with open("estimates/UCRL2-VTR(LinMix).txt", "a", encoding='utf-8') as fw:
                    # Convert theta_k to a string, replacing any newlines with a space
                    theta_k_str = str(theta_k).replace('\n', ' ')
                    fw.write(' '.join([theta_k_str, str(self.env.timestep)]) + '\n')

                u_k = self.perform_value_iteration()
                w_k = u_k - (np.max(u_k) + np.min(u_k)) / 2
                pi_k = self.derive_policy(u_k)
                total_reward += self.execute_policy(pi_k, w_k, pbar)

            # Update prev_t_k to the start time of the current episode for the next iteration
            prev_t_k = t_k

        return total_reward

    def update_confidence_set_constraint(self, theta_k, beta_k, Sigma_k):
        '''
        Update the quadratic confidence set constraint: ||Sigma_k^{1/2}(theta - theta_k)||_2 <= beta_k.
        This is done once per episode, so the constraints do not need to be modified repeatedly.
        '''
        # Confidence set constraint: ||Sigma_k^{1/2}(theta - theta_k)||_2 <= beta_k
        diff_theta = self.theta_var - theta_k

        # Use gp.QuadExpr to build the quadratic constraint ||Sigma_k^{1/2}(theta - theta_k)||_2^2
        Sigma_sqrt = np.linalg.cholesky(Sigma_k)
        quadratic_term = (Sigma_sqrt @ diff_theta) @ (Sigma_sqrt @ diff_theta)

        # Remove the previous confidence set constraint if it exists
        if hasattr(self, 'confidence_set_constraint'):
            self.gurobi_model.remove(self.confidence_set_constraint)

        # Add the new quadratic constraint
        self.confidence_set_constraint = self.gurobi_model.addConstr(quadratic_term <= beta_k**2, name="confidence_set")

        # Warm-start optimization with the current theta_k (reshape needed to be a 1-D array)
        self.theta_var.start = theta_k.flatten()

    def initialize_episode_params(self, prev_t_k):
        '''Initialize parameters for each episode.'''
        t_k = self.env.timestep
        theta_k = np.copy(np.linalg.solve(self.Sigma, self.b))
        beta_k = self.D * np.sqrt(self.d * np.log((self.lambda_reg + t_k * self.D) / (self.delta * self.lambda_reg))) + np.sqrt(self.lambda_reg) * self.B
        Sigma_k = np.copy(self.Sigma)

        prev_epi_size = t_k - prev_t_k
        if prev_epi_size <= 10:
            N_k = 41
        elif prev_epi_size > self.T - self.env.timestep:
            N_k = prev_epi_size 
        else:
            N_k = prev_epi_size * 4
            
        return t_k, theta_k, beta_k, Sigma_k, N_k

    def derive_policy(self, u_k):
        '''Derive the optimal policy based on the updated Q-values.'''
        q = np.zeros((self.env.nState, self.env.nAction))
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                theta_k_sa = self.solve_max_theta_single_sa(s, a, u_k)
                phi_u_k_sa = self.compute_phi_F(s, a, u_k)
                q[s, a] = self.env.reward[s, a] + np.dot(theta_k_sa, phi_u_k_sa)
                
        policy = np.argmax(q, axis=1)
        return policy

    def perform_value_iteration(self):
        '''Perform value iteration and return the updated Q and V values.'''
        u = np.zeros(self.env.nState)
        epsilon = 1e-4
        with tqdm(desc="Value Iteration (UCRL2-VTR)", leave=False) as pbar:
            while True:
                prev_u = u.copy()
                u = self.value_iteration(prev_u)
                diff_u = np.max(u - prev_u) - np.min(u - prev_u)
                if diff_u < epsilon:
                    break
                pbar.update(1)
        return u

    def value_iteration(self, prev_u):
        '''Perform value iteration for a single step.'''
        u = np.zeros(self.env.nState)
        q = np.zeros((self.env.nState, self.env.nAction))
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                q[s, a] = self.compute_Q(s, a, prev_u)
            u[s] = np.max(q[s, :])
        return u

    def compute_Q(self, s, a, u):
        '''
        Compute the Q-value using the solution to the maximization problem.
        '''
        # Solve for the optimal theta in the confidence set
        theta_star = self.solve_max_theta_single_sa(s, a, u)

        # Compute the transition probabilities using theta_star
        probabilities = np.dot(self.phi[s, a, :], theta_star)
        probabilities /= np.sum(probabilities)  # Ensure that probabilities sum to 1

        # Compute the Q-value as the reward plus the expected value of the next states
        Q_value = self.env.reward[s, a] + self.gamma * np.dot(probabilities, u)

        return Q_value

    def solve_max_theta_single_sa(self, s, a, u):
        '''
        Solve the maximization problem for a single (s,a) using Gurobi.
        '''
        # Compute phi_u for the specific (s,a)
        phi_u = self.compute_phi_F(s, a, u)  # shape (d,)

        # Update the Gurobi model objective function
        self.gurobi_model.setObjective(phi_u @ self.theta_var, GRB.MAXIMIZE)

        # Optimize the model
        self.gurobi_model.optimize()

        # Check if a solution is found
        if self.gurobi_model.status == GRB.OPTIMAL:
            theta_star = self.theta_var.X  # Extract the optimal solution
        else:
            raise ValueError("Optimization did not converge")

        return theta_star

    def execute_policy(self, pi_k, w_k, pbar):
        '''Execute the learned policy and update rewards and parameters.'''
        total_reward = []
        Sigma_k = np.copy(self.Sigma)
        i = 0
        while not self.terminate_episode(Sigma_k):
            s_t = self.env.state
            a_t = pi_k[s_t]

            s_next, reward = self.env.step(s_t, a_t)
            total_reward.append(reward)

            phi_W = self.compute_phi_F(s_t, a_t, w_k)

            self.Sigma += np.outer(phi_W, phi_W)
            self.b += phi_W * w_k[s_next]
            i += 1
            pbar.update(1)
        return total_reward

    def terminate_episode(self, Sigma_k):
        '''Check if the episode should terminate based on Gram matrix determinant.'''
        return np.linalg.det(self.Sigma) > 2 * np.linalg.det(Sigma_k) or self.env.timestep >= self.env.T

    def compute_phi_F(self, s_t, a_t, F):
        '''Compute the feature map \phi_{W_t}(s_t, a_t).'''
        return np.sum(self.phi[s_t, a_t, :] * F[:, np.newaxis], axis=0)


import numpy as np

np.random.seed(0)
class LinearMixtureMDP:
    def __init__(self, d, nState, nAction, theta_star, gamma, T):
        '''
        Initialize a Linear Mixture MDP with discount factor.
        Args:
            d - int - dimensionality of the feature vector
            nState - int - number of states
            nAction - int - number of actions
            theta_star - np.array - true parameter for the linear transition model
            gamma - float - discount factor
            T - int - episode length (time horizon)
        '''
        self.d = d
        self.nState = int(nState)
        self.nAction = int(nAction)
        self.theta_star = theta_star
        self.gamma = gamma
        self.T = int(T)
        self.state = 0
        self.timestep = 0
        self.phi = self.generate_phi()  # Generate phi during initialization
        self.reward = self.generate_reward()  # Generate deterministic reward function

        # Compute q_star, v_star, J_star, and H during initialization
        self.q_star, self.v_star, self.J_star = self.compute_q_star_average_reward()
        self.H = self.compute_upper_bound_H()  # Compute H based on the computed q^* and v^*

    def reset(self):
        '''Reset the environment to the initial state.'''
        self.state = 0
        self.timestep = 0
        return self.state

    def generate_phi(self):
        '''Generates and returns the feature map phi(s, a, s') that ensures the probability constraint.'''
        np.random.seed(0)
        phi = np.zeros((self.nState, self.nAction, self.nState, self.d))  # Ensure dimensions are integers
        
        for s in range(self.nState):
            for a in range(self.nAction):
                for s_prime in range(self.nState):
                    phi[s, a, s_prime] = np.random.rand(self.d)
                
                # Normalize the feature vector to ensure valid transition probabilities
                dot_products = np.array([np.dot(phi[s, a, s_prime], self.theta_star) for s_prime in range(self.nState)])
                normalization_factor = np.sum(dot_products)
                if normalization_factor > 0:
                    for s_prime in range(self.nState):
                        phi[s, a, s_prime] *= 1 / normalization_factor
        
        return phi

    def generate_reward(self):
        '''Generates a deterministic reward function r(s, a) in [0, 1].'''
        reward = np.random.rand(self.nState, self.nAction) * 10 # Generate random values in [0, 1]
        return reward

    def transition_prob(self, s, a):
        '''Compute transition probabilities directly using phi(s, a) and theta_star without post-normalization.'''
        probs = np.array([np.dot(self.phi[s, a, s_prime], self.theta_star) for s_prime in range(self.nState)])
        assert np.all(probs >= 0), "Transition probabilities should be non-negative"
        assert np.isclose(np.sum(probs), 1), "Transition probabilities should sum to 1"
        return probs

    def step(self, s, a):
        '''Take one step in the environment.'''
        transition_probs = self.transition_prob(s, a)
        newState = np.random.choice(self.nState, p=transition_probs)
        reward = self.reward[s, a]  # Deterministic reward
        self.state = newState
        self.timestep += 1
        return newState, reward

    def compute_q_star_average_reward(self):
        '''
        Compute the optimal action-value function q^*(s, a) using the Bellman optimality equation for the average reward setting.
        '''
        q_star = np.zeros((self.nState, self.nAction))  # Initialize q^*
        v_star = np.zeros(self.nState)  # Bias function (v^*)
        J_star = 0  # Initialize average reward
        epsilon = 1e-6  # Convergence threshold
        max_iterations = 1000

        for _ in range(max_iterations):
            v_new = np.zeros(self.nState)
            J_new = J_star
            for s in range(self.nState):
                for a in range(self.nAction):
                    transition_probs = self.transition_prob(s, a)
                    expected_value = np.sum([transition_probs[s_prime] * v_star[s_prime] for s_prime in range(self.nState)])
                    q_star[s, a] = self.reward[s, a] + expected_value  # Compute q^*(s, a)
                v_new[s] = np.max(q_star[s, :]) - J_star  # Update bias function using q^*
            J_new = np.mean([q_star[s, np.argmax(q_star[s, :])] for s in range(self.nState)])  # Update average reward

            if np.max(np.abs(v_new - v_star)) < epsilon:
                break
            v_star = v_new
            J_star = J_new

        return q_star, v_star, J_star

    def compute_upper_bound_H(self):
        '''Compute the upper bound H = 2 * sp(v^*) using q^*(s, a) in the average reward setting.'''
        span_v_star = np.max(self.v_star) - np.min(self.v_star)  # Compute the span of v^*
        return 2 * span_v_star

    def run_optimal_policy_from_q_star(self):
        '''Run the optimal policy based on q^* obtained from compute_q_star_average_reward.'''
        optimal_policy = np.argmax(self.q_star, axis=1)  # Derive optimal policy from q^*
        total_reward_optimal = []  # To store the rewards obtained by the optimal policy

        self.reset()  # Reset environment to the initial state
        for t in range(self.T):
            s_t = self.state
            a_t = optimal_policy[s_t]  # Select the action from the optimal policy
            _, reward = self.step(s_t, a_t)  # Take the action and receive reward
            total_reward_optimal.append(reward)

        return total_reward_optimal

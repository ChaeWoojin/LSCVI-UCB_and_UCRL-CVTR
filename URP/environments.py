import numpy as np

class LinearMixtureMDP:
    def __init__(self, d, nState, nAction, theta_star, gamma, T):
        '''
        Initialize a Linear Mixture MDP with discount factor.
        Args:
            d - int - dimensionality of the feature vector
            nState - int - number of states
            nAction - int - number of actions
            T - int - episode length (time horizon)
            theta_star - np.array - true parameter for the linear transition model
            gamma - float - discount factor
        '''
        self.d = d  # Dimensionality of the feature vector
        self.nState = int(nState)  # Number of states (ensure it's an integer)
        self.nAction = int(nAction)  # Number of actions (ensure it's an integer)
        self.theta_star = theta_star
        self.gamma = gamma
        self.T = int(T)  # Episode length or time horizon
        self.state = 0
        self.timestep = 0
        self.phi = self.generate_phi()  # Generate phi upon initialization
        self.reward = self.generate_reward()  # Initialize deterministic reward function

    def reset(self):
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

                dot_products = np.array([np.dot(phi[s, a, s_prime], self.theta_star) for s_prime in range(self.nState)])
                normalization_factor = np.sum(dot_products)
                if normalization_factor > 0:
                    for s_prime in range(self.nState):
                        phi[s, a, s_prime] *= 1 / normalization_factor
        
        return phi

    def generate_reward(self):
        '''Generates a deterministic reward function r(s, a) in [0, 1].'''
        np.random.seed(0)
        reward = np.random.rand(self.nState, self.nAction)  # Generate random values in [0, 1]
        return reward
    
    def transition_prob(self, s, a):
        '''Compute transition probabilities directly using phi(s, a) and theta_star without post-normalization.'''
        probs = np.array([np.dot(self.phi[s, a, s_prime], self.theta_star) for s_prime in range(self.nState)])
        assert np.all(probs >= 0), "Transition probabilities should be non-negative"
        assert np.isclose(np.sum(probs), 1), "Transition probabilities should sum to 1"
        return probs

    def step(self, s, a):
        '''
        Takes one step in the environment.
        Args:
            s - int - current state
            a - int - action taken
        Returns:
            newState - int - next state based on the transition model
            reward   - float - reward for the step
        '''
        transition_probs = self.transition_prob(s, a)
        newState = np.random.choice(self.nState, p=transition_probs)
        reward = self.reward[s, a]  # Deterministic reward
        self.state = newState
        self.timestep += 1
        return newState, reward

    def compute_q_star_average_reward(self):
        '''
        Compute the optimal action-value function q^*(s, a) using Bellman optimality equation for average reward setting:
        J^* + q^*(s,a) = r(s,a) + \mathbb{P}v^*(s)
        '''
        q_star = np.zeros((self.nState, self.nAction))  # Initialize q^* to zero
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
        _, v_star, _ = self.compute_q_star_average_reward()  # Compute the optimal q^*, v^*, and J^*
        span_v_star = np.max(v_star) - np.min(v_star)  # Compute the span of v^*
        return 2 * span_v_star

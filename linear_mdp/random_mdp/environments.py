import numpy as np

class LinearMDP:
    def __init__(self, d, nState, nAction, gamma, T):
        '''
        Initialize a Linear MDP with discount factor.
        Args:
            d - int - dimensionality of the feature vector
            nState - int - number of states
            nAction - int - number of actions
            gamma - float - discount factor
            T - int - episode length (time horizon)
        '''
        self.d = d  # Dimensionality of the feature vector
        self.nState = int(nState)  # Number of states
        self.nAction = int(nAction)  # Number of actions
        self.gamma = gamma  # Discount factor
        self.T = int(T)  # Episode length
        self.state = 0  # Initialize starting state
        self.timestep = 0  # Initialize timestep
        self.mu = self.generate_mu()  # True parameter matrix for the transition kernel (nState x d)
        self.phi = self.generate_phi()  # Generate feature map for state-action pairs
        self.theta = self.generate_theta()  # Generate theta based on phi to ensure reward within [0, 1]
        self.reward = self.generate_reward()  # Generate reward based on phi and theta  
        self.q_star, self.v_star, self.J_star = self.compute_q_star_average_reward()  # Compute q*, v*, and J*
        self.H = self.compute_upper_bound_H()  # Compute H during initialization

    def reset(self):
        '''Reset the environment to the initial state.'''
        self.state = 0
        self.timestep = 0
        return self.state

    def generate_mu(self):
        '''Generate the feature map mu(s) for every state.'''
        np.random.seed(0)
        mu = np.zeros((self.nState, self.d))
        for s in range(self.nState):
            mu[s] = np.random.rand(self.d)
        
        return mu

    def generate_phi(self):
        '''Generate the feature map φ(s, a) for each state-action pair.'''
        np.random.seed(0)
        phi = np.zeros((self.nState, self.nAction, self.d))  # Feature map φ(s, a) ∈ R^d

        for s in range(self.nState):
            for a in range(self.nAction):
                # Randomly generate the feature vector φ(s, a)
                phi[s, a] = np.random.rand(self.d)

                # Ensure the sum of inner products φ(s,a)⋅μ(s') over s' is 1
                dot_products = np.array([np.dot(phi[s, a], self.mu[s_prime]) for s_prime in range(self.nState)])
                normalization_factor = np.sum(dot_products)

                if normalization_factor > 0:
                    phi[s, a] /= normalization_factor  # Normalize φ(s, a) so that the sum of inner products equals 1

        return phi

    def generate_theta(self):
        '''Generate the true parameter θ after generating φ, ensuring rewards are within [0, 1].'''
        theta = np.random.rand(self.d)  # Start by generating a random θ
        
        # Adjust θ to ensure rewards are within [0, 1]
        max_inner_product = np.max([np.dot(self.phi[s, a], theta) for s in range(self.nState) for a in range(self.nAction)])
        
        if max_inner_product > 0:
            theta /= max_inner_product  # Scale θ so that the maximum reward is 1

        return theta

    def generate_reward(self):
        '''Generate a deterministic reward function r(s, a) as the inner product of φ(s, a) and θ.'''
        reward = np.zeros((self.nState, self.nAction))  # Initialize reward array
        for s in range(self.nState):
            for a in range(self.nAction):
                # Compute the inner product between φ(s,a) and θ
                reward[s, a] = np.dot(self.phi[s, a], self.theta)

        return reward

    def transition_prob(self, s, a):
        '''Compute the transition probabilities based on φ(s, a) and μ(s').'''
        probs = np.array([np.dot(self.mu[s_prime], self.phi[s, a]) for s_prime in range(self.nState)])  # μ is (nState, d) and φ(s, a) is (d,)
        assert np.all(probs >= 0), "Transition probabilities should be non-negative"
        assert np.isclose(np.sum(probs), 1), "Transition probabilities should sum to 1"
        return probs

    def step(self, s, a):
        '''Take one step in the environment.'''
        transition_probs = self.transition_prob(s, a)
        newState = np.random.choice(self.nState, p=transition_probs)
        reward = self.reward[s, a]
        
        self.state = newState
        self.timestep += 1
        
        return newState, reward

    def compute_upper_bound_H(self):
        '''Compute the upper bound H = 2 * sp(v^*).'''
        span_v_star = np.max(self.v_star) - np.min(self.v_star)  # Compute the span of v^*
        return 2 * span_v_star

    def compute_q_star_average_reward(self):
        '''Compute the optimal action-value function q^*(s, a) using Bellman optimality equation for average reward setting.'''
        q_star = np.zeros((self.nState, self.nAction))  # Initialize q^*
        v_star = np.zeros(self.nState)  # Bias function v^*
        J_star = 0  # Average reward
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
                v_new[s] = np.max(q_star[s, :]) - J_star  # Update bias function
            J_new = np.mean([q_star[s, np.argmax(q_star[s, :])] for s in range(self.nState)])  # Update average reward

            if np.max(np.abs(v_new - v_star)) < epsilon:
                break
            v_star = v_new
            J_star = J_new

        return q_star, v_star, J_star

    def run_optimal_policy(self):
        '''Run the environment using the optimal policy based on the computed q^*.'''
        optimal_policy = np.argmax(self.q_star, axis=1)  # Derive optimal policy from q^*
        total_reward_optimal = []  # Store rewards

        self.reset()  # Reset environment
        for t in range(self.T):
            s_t = self.state
            a_t = optimal_policy[s_t]  # Take action from optimal policy
            _, reward = self.step(s_t, a_t)  # Take the action and receive reward
            total_reward_optimal.append(reward)

        return total_reward_optimal

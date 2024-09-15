import numpy as np
import itertools

class HardLinearMDP:
    def __init__(self, d, H, T):
        self.d = d
        self.H = H
        self.T = T
        self.delta = 1 / H
        self.nState = 2
        self.nAction = 2**(self.d - 1)
        self.timestep = 0
        
        self.state = 0  # initial state is x0
        
        self.triangle = ((1/45) * np.sqrt(2 * np.log(2) / 5) * self.d) / (self.H * self.T)
        self.alpha = np.sqrt(self.triangle / ((d - 1) * (self.triangle + 1)))         
        self.beta =  np.sqrt(1 / (1 + self.triangle))
        
        self.theta = np.random.choice([-1, 1], self.d - 1) * self.triangle / (d-1)
        self.theta_tilde = np.concatenate((self.theta / self.alpha, np.array([1 / self.beta])))
        self.actions = np.array(list(itertools.product([-1, 1], repeat=d-1)))
        
        self.phi = self.generate_phi()
        self.mu = self.generate_mu()
        self.reward = self.generate_reward()
        
        self.J_star = (self.delta + self.triangle) / (2 * self.delta + self.triangle)
        self.H = 10  # Upper bound H; can be computed from q*
    
    def generate_phi(self):
        """ Feature map for (state, action) pairs. Maps to a d+1-dimensional vector. """
        phi = np.zeros((self.nState, self.nAction, self.d + 1))
        for i in range(self.nAction):
            phi[0, i] = np.concatenate((self.alpha * self.actions[i], np.array([self.beta, 0])))
            phi[1, i] = np.concatenate((np.zeros(self.d), np.array([1])))
        return phi
    
    def generate_mu(self):
        """ Feature map for (next_state). Maps to d+1-dimensional vector. """
        mu = np.zeros((self.nState, self.d + 1))
        mu[0] = np.concatenate((-self.theta / self.alpha, np.array([(1 - self.delta) / self.beta, self.delta])))
        mu[1] = np.concatenate((self.theta / self.alpha, np.array([self.delta / self.beta, 1 - self.delta])))
        return mu
        
    def generate_reward(self):
        """ Linear reward function: r(s, a) = <phi(s, a), theta>. """
        reward = np.zeros((self.nState, self.nAction))
        xi = np.concatenate((np.zeros(self.d), np.array([1])))
        for s in range(self.nState):
            for a in range(self.nAction):
                reward[s, a] = np.dot(self.phi[s, a], xi)
        return reward
    
    def transition_prob(self, s, a):
        """ Transition probabilities are determined by the linear feature map and theta. """
        phi_sa = self.phi[s, a]
        prob_x0 = np.dot(phi_sa, self.mu[0])
        prob_x1 = np.dot(phi_sa, self.mu[1])
        
        return np.array([prob_x0, prob_x1])
    
    def step(self, state, action):
        """ Perform a step in the MDP. """
        probs = self.transition_prob(state, action)
        reward = self.reward[state, action]
        next_state = np.random.choice([0, 1], p=probs)
        self.state = next_state
        self.timestep += 1
        return self.state, reward
    
    def run_optimal_policy(self):
        """ Simulate an episode following the optimal policy. """
        optimal_policy = self.generate_optimal_policy()
        # print(optimal_policy) 
        assert(len(optimal_policy) == self.nState)
        total_reward_optimal = []

        self.reset()
        for t in range(self.T):
            s_t = self.state
            a_t = optimal_policy[s_t]  # Select the action from the optimal policy
            _, reward = self.step(s_t, a_t)  # Take the action and receive reward
            total_reward_optimal.append(reward)

        return total_reward_optimal
    
    def generate_optimal_policy(self):
        policy = np.zeros(self.nState, dtype=int)
        for s in range(self.nState):
            for i in range(self.nAction):
                if (np.dot(self.actions[i], self.theta) == self.triangle):
                    policy[s] = i
        return policy
    
    def reset(self):
        self.state = 0
        self.timestep = 0
        return self.state

# Usage example:
# d = 8  # dimensionality
# D = 100  # big radius
# T = 1000

# mdp = HardLinearMDP(d=d, D=D, T=T)
# state = mdp.reset()

# for _ in range(T):
#     action = np.random.choice(2**(d-1), 1)[0]  # randomly choose an action
#     next_state, reward = mdp.step(action)
#     print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")

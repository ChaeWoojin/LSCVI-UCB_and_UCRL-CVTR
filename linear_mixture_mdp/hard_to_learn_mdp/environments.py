import numpy as np
import itertools

class HardLinearMixtureMDP:
    def __init__(self, d, D, T):
        self.d = d
        self.D = D
        self.T = T
        self.delta = 1 / D
        self.nState = 2
        self.nAction = 2**(self.d - 1)
        self.timestep = 0
        
        self.state = 0  # initial state is x0
        
        self.triangle = ((1/45) * np.sqrt(2 * np.log(2) / 5) * self.d) / (self.D * self.T)
        self.alpha = np.sqrt(self.triangle / ((d - 1) * (self.triangle + 1)))         
        self.beta =  np.sqrt(1 / (1 + self.triangle))
        
        self.theta = np.random.choice([-1, 1], self.d - 1) * self.triangle / (d-1)
        self.theta_tilde = np.concatenate((self.theta / self.alpha, np.array([1 / self.beta])))
        self.actions = np.array(list(itertools.product([-1, 1], repeat=d-1)))
        self.reward = self.generate_reward()
        self.phi = self.generate_phi()
        
        self.J_star = (self.delta + self.triangle) / (2 * self.delta + self.triangle)
        self.H = 10
    
    def generate_reward(self):
        reward = np.zeros((self.nState, self.nAction))
        reward[0, :] = np.zeros(self.nAction)
        reward[1, :] = np.ones(self.nAction)
        return reward
    
    def generate_phi(self):
        phi = np.zeros((2, 2**(self.d - 1), 2, self.d))
        for i in range(self.nAction):
            phi[0, i, 0] = np.concatenate((-self.alpha * self.actions[i], np.array([self.beta * (1 - self.delta)]))) 
            phi[0, i, 1] = np.concatenate((self.alpha * self.actions[i], np.array([self.beta * self.delta])))
            phi[1, i, 0] = np.concatenate((np.zeros(self.d - 1), np.array([self.beta * self.delta])))
            phi[1, i, 1] = np.concatenate((np.zeros(self.d - 1), np.array([self.beta * (1 - self.delta)])))
            print("P(0 | %d, 0) = %f"%(i, np.dot(phi[0, i, 0], self.theta_tilde)))
            print("P(0 | %d, 1) = %f"%(i, np.dot(phi[0, i, 1], self.theta_tilde)))
            print("P(1 | %d, 0) = %f"%(i, np.dot(phi[1, i, 0], self.theta_tilde)))
            print("P(1 | %d, 1) = %f"%(i, np.dot(phi[1, i, 1], self.theta_tilde)))
        print("theta_star: ", self.theta)
        return phi
    
    def transition_prob(self, s, a):
        action = self.actions[a]
        if s == 0:  # from state x0
            prob_x0 = 1 - self.delta - np.dot(action, self.theta)
            prob_x1 = self.delta + np.dot(action, self.theta)
        else:       # from state x1
            prob_x0 = self.delta
            prob_x1 = 1 - self.delta

        return np.array([prob_x0, prob_x1])
    
    def step(self, state, action):
        probs = self.transition_prob(state, action)
        reward = self.reward[state, action]
        next_state = np.random.choice([0, 1], p=probs)
        self.state = next_state
        self.timestep += 1
        return self.state, reward
    
    def run_optimal_policy(self):
        optimal_policy = self.generate_optimal_policy()  # Derive optimal policy from q^*
        assert(len(optimal_policy) == self.nState)
        total_reward_optimal = [] # To store the rewards obtained by the optimal policy

        self.reset()  # Reset environment to the initial state
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

# # Usage example:
# d = 8  # dimensionality
# D = 100  # big radius
# T = 1000

# mdp = HardMDP(d=d, D=D, T=T)
# state = mdp.reset()

# for _ in range(T):
#     action = np.random.choice(2**(d-1), 1)[0]  # randomly choose an action
#     next_state, reward = mdp.step(action)
#     print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")

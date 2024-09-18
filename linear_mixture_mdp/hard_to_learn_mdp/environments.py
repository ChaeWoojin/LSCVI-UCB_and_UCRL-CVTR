import numpy as np
import itertools

np.random.seed(0)
class HardLinearMixtureMDP:
    def __init__(self, d, D, T):
        self.d = d
        self.D = D
        self.H = self.D
        self.T = T
        self.delta = 1 / self.D
        self.nState = 2
        self.nAction = 2**(self.d - 1)
        self.timestep = 0
        self.state = 0 
        
        self.triangle = (1/45 * np.sqrt(2*np.log(2)/5)) * (self.d) / np.sqrt(self.H * self.T)
        self.alpha = np.sqrt(self.triangle / ((d - 1) * (1 + self.triangle)))         
        self.beta =  np.sqrt(1 / (1 + self.triangle))
        
        self.theta = np.random.choice([-1, 1], self.d - 1) * self.triangle / (d-1)
        self.theta_tilde = np.concatenate((self.theta / self.alpha, np.array([1 / self.beta])))
        self.actions = np.array(list(itertools.product([-1, 1], repeat=d-1)))
        self.reward = self.generate_reward()
        self.phi = self.generate_phi()
        self.varphi = self.generate_varphi()
        
        self.J_star = (self.delta + self.triangle) / (2 * self.delta + self.triangle)
        self.action_rank = np.argsort(np.array([np.dot(-self.actions[i], self.theta) for i in range(self.nAction)]))
                      
    def reset(self):
        self.state = 0
        self.timestep = 0
        return self.state
        
    def generate_reward(self):
        reward = np.zeros((self.nState, self.nAction))
        reward[0, :] = np.zeros(self.nAction)
        reward[1, :] = np.ones(self.nAction)
        return reward
    
    def generate_phi(self):
        phi = np.zeros((2, self.nAction, 2, self.d))  # (current state, action, next state, feature dimension)
        
        for i in range(self.nAction):
            action_vector = self.actions[i]  # Assuming self.actions is a list of action vectors with size (self.d - 1)

            phi[0, i, 0] = np.concatenate((-self.alpha * action_vector, [self.beta * (1 - self.delta)]))
            phi[0, i, 1] = np.concatenate((self.alpha * action_vector, [self.beta * self.delta]))
            phi[1, i, 0] = np.concatenate((np.zeros(self.d - 1), [self.beta * self.delta]))
            phi[1, i, 1] = np.concatenate((np.zeros(self.d - 1), [self.beta * (1 - self.delta)]))
            
        return phi
    
    def generate_varphi(self):
        varphi = np.zeros((self.nState, self.nAction, self.d))
        for s in range(self.nState):
            for a in range(self.nAction):
                target_value = self.reward[s, a] 
                varphi[s, a, :-1] = np.random.rand(self.d - 1)
                remaining = (target_value - np.dot(varphi[s, a, :-1], self.theta_tilde[:-1])) / self.theta_tilde[-1]
                varphi[s, a, -1] = remaining

        return varphi
    
    def transition_prob(self, s, a):
        action = self.actions[int(a)]
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
        optimal_policy = self.generate_optimal_policy() 
        assert(len(optimal_policy) == self.nState)
        total_reward_optimal = []

        self.reset() 
        for t in range(self.T):
            s_t = self.state
            a_t = optimal_policy[s_t] 
            _, reward = self.step(s_t, a_t) 
            total_reward_optimal.append(reward)
        return total_reward_optimal
    
    def generate_optimal_policy(self):
        policy = np.zeros(self.nState, dtype=int)
        for s in range(self.nState):
            for i in range(self.nAction):
                if (np.dot(self.actions[i], self.theta) == self.triangle):
                    policy[s] = i
        return policy


d = 3
D = 5
T = 10000

mdp = HardLinearMixtureMDP(d=d, D=D, T=T)
# actions = mdp.actions
# theta = mdp.theta
# delta = mdp.delta
# index = np.array([np.dot(actions[i], theta) for i in range(mdp.nAction)])
# rank = np.argsort(-index)
# print(rank)
# print(1 - delta - np.min(np.array([np.dot(actions[i], theta) for i in range(mdp.nAction)])))
# print(1 - delta - np.max(np.array([np.dot(actions[i], theta) for i in range(mdp.nAction)])))
# print(mdp.varphi)

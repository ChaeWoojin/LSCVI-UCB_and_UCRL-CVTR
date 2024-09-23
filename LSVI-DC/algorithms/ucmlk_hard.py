import numpy as np
# import gurobipy as gp
# from gurobipy import GRB
import cvxpy as cp
from scipy.optimize import fmin_tnc
from tqdm import tqdm

# import os, sys
# sys.path.append(r'C:\Users\uqpua\OneDrive\Desktop\UCRL-MNL')
# sys.path.append(r'/Users/optalab/Documents/UCRL-MNL/')
# print(os.getcwd())
# print(sys.path)
# print(os.listdir())
# from env import *

class UCMLK:
    def __init__(self, env, c, gamma, N, eta, T):
        self.env = env
        # self.K = K  # number of episodes
        self.c = c  # confidence radius parameter
        self.gamma = gamma  # discount factor
        self.N = N  # number of DEVI rounds
        self.eta = eta
        self.T = T  # number of rounds

        # Feature dimensions
        self.d1 = self.env.nState * self.env.nAction
        self.d2 = self.env.nState
        self.d = self.d1 * self.d2

        # # reachable states
        self.reachable_states = {(s,a):{} for s in self.env.states.keys() for a in range(self.env.nAction)}

        for a in range(self.env.nAction):
            self.reachable_states[(0,a)] = {0, 1}
            self.reachable_states[(1,a)] = {0, 1}
            
       # support state
        self.support_states = {(s,a):list(self.reachable_states[(s,a)])[0] for s in self.env.states.keys() for a in range(self.env.nAction)}

        # Initialize features
        self.phi = self._initialize_phi()
        self.psi = self._initialize_psi()
        self.varphi = self._initialize_varphi()

        # Initialize Q-values
        self.Q = {(s,a): 0.0 for n in range(self.N) for s in self.env.states.keys() for a in range(self.env.nAction)}
        # self.Q = {(h,s,a): 0,0 for h in range(self.env.epLen) for s in self.env.states.keys() for a in range(self.env.nAction)}

        # Initialize model parameters
        self.A = np.identity(self.d)
        self.Ainv = np.linalg.inv(self.A)
        self.lam = 1
        self.theta = np.zeros(self.d)

        # For MNL regression
        self.mnl = OMD()
        self.X = []
        self.Y = []

    def _initialize_phi(self):
        phi = {(s,a): np.zeros(self.d1) for s in self.env.states.keys() for a in range(self.env.nAction)}
        i = 0
        for key in phi.keys():
            phi[key][i] = 1
            i += 1
        return phi

    def _initialize_psi(self):
        psi = {s: np.zeros(self.d2) for s in self.env.states.keys()}
        j = 0
        for key in psi.keys():
            psi[key][j] = 1
            j += 1
        return psi

    def _initialize_varphi(self):
        varphi = {(s,a,s_): np.zeros(self.d) for s in self.psi.keys() for a in range(self.env.nAction) for s_ in self.reachable_states[s,a]}
        for s in self.psi.keys():
            for a in range(self.env.nAction):
                for s_ in self.reachable_states[(s,a)]:
                    varphi[(s,a,s_)] = np.outer(self.phi[(s,a)],
                                                 (self.psi[s_] - self.psi[self.support_states[s,a]])).flatten()
        return varphi

    def proj(self, x, lo, hi):
        return max(min(x, hi), lo)

    def DEVI(self, t_k):
        """
        Discounted Extended Value Iteration Algorithm
        
        Inputs:
        - gamma: discount factor
        - P: confidence polytope
        - N: number of rounds
        
        Returns:
        - Q: Action-value function
        """
        # Initialize
        Q = {(s, a): 1/(1 - self.gamma) for s in self.env.states.keys() for a in range(self.env.nAction)}
        V = {s: 1/(1 - self.gamma) for s in self.env.states.keys()}
        # Round
        for n in range(self.N):
            V_prev = V.copy()

            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    p = cp.Variable(len(self.reachable_states[s,a]), nonneg=True)
                    objective = cp.Maximize(cp.sum(cp.multiply(p, [V_prev[s_next] for s_next in self.reachable_states[s,a]])))
                    
                    comp = [np.exp(np.dot(self.varphi[(s,a,s_)], self.theta)) for s_ in self.reachable_states[(s,a)]]
                    p_hat = comp / np.sum(comp)
                
                    # sigma_norm = [np.sqrt(np.dot(np.dot(self.varphi[(s,a,s_)], self.Ainv), self.varphi[(s,a,s_)])) for s_ in self.reachable_states[(s,a)]] 
                    sigma_norm = [np.linalg.norm(np.dot(np.sqrt(self.Ainv), self.varphi[(s, a, s_)])) for s_ in self.reachable_states[(s, a)]]
                    R_t = max(1, 2*self.Beta(t_k)*np.max(sigma_norm))
                    R_t = min(R_t, len(self.reachable_states[(s,a)]))
                    # cp.sum([p[s_next] for s_next in self.reachable_states[(s,a)]]) == 1
                    constraints = [cp.sum([p[s_next] for s_next in range(len(self.reachable_states[(s,a)]))]) == 1,
                                   p >= 0,
                                   p <= 1,
                                   cp.sum([cp.abs(p[s_next] - p_hat[s_next]) for s_next in range(len(self.reachable_states[(s,a)]))]) <= R_t]
                    prob = cp.Problem(objective, constraints)
                    try:
                        prob.solve()
                        # print(prob.status, cp.OPTIMAL)
                        if prob.status == cp.OPTIMAL:
                            # Calculate Q-value
                            Q[s, a] = self.env.R[s,a][0] + self.gamma * prob.value
                            # print(f"s: {s}, a: {a}")
                        else:
                            print(f"Optimization failed for state {s}, action {a}. Status: {prob.status}")
                    except cp.error.SolverError:
                        print(f"Solver error for state {s}, action {a}")

                    # # Create a Gurobi model
                    # model = gp.Model("DEVI_optimization")
                    # model.setParam('OutputFlag', 0)  # Suppress Gurobi output
                    

                    # # Create variables
                    # p = model.addVars(self.env.nState, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="p")

                    # # Set objective
                    # obj = gp.quicksum(p[s_next] * V_prev[s_next] for s_next in self.reachable_states[(s,a)])
                    # model.setObjective(obj, GRB.MAXIMIZE)
                    
                    # # Add constraints from the confidence polytope P
                    # model.addConstr(gp.quicksum(p[s_next] for s_next in self.reachable_states[(s,a)]) == 1, "prob_sum")
                    # model.addConstr(gp.quicksum(gp.abs_(p[s_next] - p_hat[s_next]) for s_next in self.reachable_states[(s,a)]) <= R_t)
                    
                    # # Optimize the model
                    # model.optimize()
                    
                    # if model.status == GRB.OPTIMAL:
                    #     # Calculate Q-value
                    #     Q[s, a] = self.env.R[s,a][0] + self.gamma * model.objVal
                    # else:
                    #     print(f"Optimization failed for state {s}, action {a}")

            if n == self.N-1:
                continue

            for s in self.env.states.keys():
                V[s] = max(np.array([Q[s, a] for a in range(self.env.nAction)]))
        self.Q = Q.copy()

    def Beta(self, t_k):
        return self.c * np.sqrt(self.d) * np.log((self.env.nState * t_k / (1 -self.gamma)))**2    # delta = 1 - gamma 

    def act(self, s):
        return self.env.argmax(np.array([self.Q[(s,a)] for a in range(self.env.nAction)]))
    
    def update_theta(self, X, Y):
        self.mnl.fit(X, Y, self.theta, self.eta, self.Ainv, self.A) # x, y, theta, eta, Ainv, A
        self.theta = self.mnl.w
    
    def compute_prob(self, theta, x):  # l_t (theta)
        probs = []
        for i in range(len(x)):
            means = np.dot(x[i], theta)
            u = np.exp(means)
            logSumExp = u.sum()
            prob = u/logSumExp
            probs.append(prob)
        return probs
    
    def update_gram_matrix(self, X):
        probs = self.compute_prob(self.theta, X)

        for i in range(len(X)):
            # print(probs[i], np.outer(X[i], X[i]))
            self.A += probs[i] * np.outer(X[i], X[i])

            for j in range(len(X)):
                self.A -= probs[i] * probs[j] * np.outer(X[i], X[j])
                # print(probs[i], probs[j], )
        self.Ainv = np.linalg.inv(self.A)

    def run(self):
        print("UCMLK")
        episode_return = []  # round_return

        A_k = self.A.copy()
        # A_k = self.A
        t_k = 1
        R = 0  # cumulative reward
        for t in tqdm(range(1, self.T+1)):
            # self.env.reset()
            
            if np.linalg.det(self.A) > 2*np.linalg.det(A_k):
                t_k = t
                # A_k = self.A
                A_k = self.A.copy()
                self.DEVI(t_k)
            
            s = self.env.state
            a = self.act(s)

            # prob_tmp = {}
            # for i in self.env.states.keys():
            #     if i in self.reachable_states[s,a]:
            #         prob_tmp[i] = np.exp(np.dot(self.varphi[(s,a,i)], self.theta))
            # sum_tmp = sum(prob_tmp.values())
            # for key, value in prob_tmp.items():
            #     self.env.P[s, a][key] = value / sum_tmp
            # print(s, a, self.env.P[s, a])
            
            r, s_= self.env.advance(a)
            R += r
            # print(f't: {t}, R: {R}, s: {s}, a: {a}')
            X = []
            Y = []
            y = np.zeros(len(self.reachable_states[s,a]))
            for i in range(len(self.reachable_states[s,a])):
                    if list(self.reachable_states[s,a])[i] == s_:
                        y[i] = 1
                    X.append(self.varphi[(s,a,list(self.reachable_states[s,a])[i])])
                               
            self.Y.append(y)
            self.X.append(np.array(X))
            
            # 여기가 문제
            self.update_theta(self.X, self.Y)
            self.update_gram_matrix(X)
            episode_return.append(R)

        return episode_return

class OMD:
    # def __init__(self, theta):
    #     self.initial_theta = theta

    def compute_prob(self, theta, x):  # l_t (theta)
        probs = []
        for i in range(len(x)):
            means = np.dot(x[i], theta)
            u = np.exp(means)
            logSumExp = u.sum()
            prob = u/logSumExp
            probs.append(prob)
        return probs

    def cost_function(self, theta, x, y, eta, Ainv, A):
        m = len(x)
        prob = self.compute_prob(theta, x)
        eps = [prob[i] - y[i] for i in range(len(prob))]
        
        grad = np.zeros(len(theta))
        for i in range(len(eps)):
            for j in range(len(eps[i])):
                grad += eps[i][j] * x[i][j]                
        grad = (1/m)*grad # 다 더했으니깐 나누는거구나
        # theta_prime = theta - eta * Ainv * grad
        theta_prime = theta - eta * np.dot(Ainv, grad)

        # return np.sqrt( np.dot( np.dot((theta - theta_prime), A), (theta - theta_prime)))
        return np.dot( np.dot((theta - theta_prime), A), (theta - theta_prime))

    def gradient(self, theta, x, y, eta, Ainv, A):  # 비용함수의 그래디언트
        m = len(x)
        prob = self.compute_prob(theta, x)
        eps = [prob[i] - y[i] for i in range(len(prob))]
        
        grad = np.zeros(len(theta))
        for i in range(len(eps)):
            for j in range(len(eps[i])):
                grad += eps[i][j] * x[i][j]                
        grad = (1/m)*grad # 다 더했으니깐 나누는거구나
        theta_prime = theta - eta * np.dot(Ainv, grad)
        # theta_prime = theta - eta * Ainv * grad
        # return np.sqrt( np.dot( np.dot((theta - theta_prime), A), (theta - theta_prime)))
        return np.dot(A, (theta - theta_prime))

    def fit(self, x, y, theta, eta, Ainv, A):
        # initial_theta = theta
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, messages=0, args=(x, y, eta, A, Ainv))
        self.w = opt_weights[0]
        return self


# env = make_riverSwim(T=100, nState=6)
# env = make_hardToLearnMDP(T=100)
# agent = UCMLK(env, c=1e-2, gamma=0.9, N=10, eta=0.1 , T=100)
# agent.run()
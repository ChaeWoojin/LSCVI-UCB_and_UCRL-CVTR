import numpy as np
from scipy.optimize import fmin_tnc
from tqdm import tqdm

import os, sys
# sys.path.append(r'C:\Users\uqpua\OneDrive\Desktop\UCRL-MNL')
sys.path.append(r'/Users/optalab/Documents/UCRL-MNL/')
# print(os.getcwd())
# print(sys.path)
# print(os.listdir())
from env2 import *

class UCRL_MNL(object):
    
    def __init__(self, env, K, c):
        
        self.env = env
        # the number of episodes
        self.K = K        
        # d_1
        self.d1 = self.env.nState * self.env.nAction        
        # d_2
        self.d2 = self.env.nState
        # d
        self.d = self.d1 * self.d2
        # phi features
        self.phi = {(s,a): np.zeros(self.d1) for s in self.env.states.keys() for a in range(self.env.nAction)}
        i = 0
        for key in self.phi.keys():
            self.phi[key][i] = 1
            i += 1
            
        # psi features
        self.psi = {(s): np.zeros(self.d2) for s in self.env.states.keys()}
        j = 0
        for key in self.psi.keys():
            self.psi[key][j] = 1
            j += 1
        
        # reachable states
        self.reachable_states = {(s,a):{} for s in self.env.states.keys() for a in range(self.env.nAction)}
        for s in self.env.states.keys():
            self.reachable_states[(s,0)] = {max(s-1, 0)}
            self.reachable_states[(s,1)] = {s, min(s+1, self.env.nState-1), max(0, s-1)}
        
        # support state
        self.support_states = {(s,a):list(self.reachable_states[(s,a)])[0] for s in self.env.states.keys() for a in range(self.env.nAction)}
        
        ############################################################################################################
        ###### Revision 
        # xi features (old-fashioned)
#         self.xi = {(s,a,s_): np.zeros(self.d1*self.d2) for s in self.psi.keys() for a in range(self.env.nAction) \
#                                  for s_ in self.psi.keys()}
#         for s in self.psi.keys():
#             for a in range(self.env.nAction):
#                 for s_ in self.psi.keys():
#                     self.xi[(s,a,s_)] = np.outer(self.phi[(s,a)],
#                                                  (self.psi[s_] - self.psi[0])).flatten()
        ##
        # varphi features (new)
        self.varphi = {(s,a,s_): np.zeros(self.d1*self.d2) for s in self.psi.keys() for a in range(self.env.nAction) \
                                 for s_ in self.reachable_states[s,a]}
        for s in self.psi.keys():
            for a in range(self.env.nAction):
                for s_ in self.reachable_states[s,a]:
                    self.varphi[(s,a,s_)] = np.outer(self.phi[(s,a)],
                                                 (self.psi[s_] - self.psi[self.support_states[s,a]])).flatten()
        ############################################################################################################
        
        
        # Initialize our Q matrix
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        
        # gram matrix
        self.A = np.identity(self.d1*self.d2)
        self.Ainv = np.linalg.inv(self.A)
        self.lam = 1

        # theta
        self.theta = np.zeros(self.d1*self.d2)
        
        # confidence raidus param
        self.c = c
        
        # for MNL regression
#         self.mnl = RegularizedMNLRegression()
#         self.X = np.zeros((self.env.nState, self.d1*self.d2))[np.newaxis, ...]
#         self.Y = np.zeros(self.env.nState)[np.newaxis, ...]
        self.mnl = RegularizedMNLRegression_sep()
        self.X = []
        self.Y = []

        
        
        
    def act(self, s, h):
        """
        a function that returns the argmax of Q given the state and timestep
        """

        return self.env.argmax(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
    
        
    def proj(self, x, lo, hi):
        '''Projects the value of x into the [lo,hi] interval'''
        
        return max(min(x, hi), lo)
    
    def compute_Q(self, k):
        """
        a function that computes the Optimistic Q-values, see step 6 and Eq 4 and 8
        """
        Q = {(h, s, a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys()
        for a in range(self.env.nAction)}
        V = {h: np.zeros(self.env.nState) for h in range(self.env.epLen + 1)}
           
        for h in range(self.env.epLen-1, -1, -1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    
                    r = self.env.R[s,a][0]
#                     comp = [] # list of exp(varphi^T theta)
                    comp = [np.exp(np.dot(self.varphi[(s,a,s_)], self.theta)) for s_ in self.reachable_states[(s,a)]] 
#                     w_norm = [] # list of weighted norm of varphi
                    w_norm = [np.sqrt(np.dot(np.dot(self.varphi[(s,a,s_)], self.Ainv), self.varphi[(s,a,s_)])) for s_ in self.reachable_states[(s,a)]] 
#                     next_value = [] # list of Vhat[h+1][s_]
                    next_value = [V[h+1][s_] for s_ in self.reachable_states[(s,a)]] 
                    ############################################################################################################
                    ###### Revision                
#                     for s_ in self.reachable_states[(s,a)]:
#                         comp.append(np.exp(np.dot(self.varphi[(s,a,s_)], self.theta)))
#                         w_norm.append(np.sqrt(np.dot(np.dot(self.varphi[(s,a,s_)], self.Ainv), self.varphi[(s,a,s_)])))
#                         next_value.append(V[h+1][s_])
                        
                    val = (np.dot(comp, next_value)/np.sum(comp)) + 2*self.env.epLen*self.Beta(k)*np.max(w_norm)
                    Q[h, s, a] = self.proj(r + val, 0, self.env.epLen)
                V[h][s] = max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
        self.Q = Q.copy()    
        
    def Beta(self, k):
        return self.c*np.sqrt(self.d1*self.d2*np.log(1+((k*self.env.epLen*self.env.nState)/(self.d1*self.d2*self.lam))))

    def update_theta(self, k, X, Y):
        self.mnl.fit(X, Y, self.theta, self.lam)
        self.theta = self.mnl.w
        
#     def update_gram_matrix(self, X):
#         print("Original")
# #         for i in range(len(X)):
# #             for j in range(len(X[i])):
# #                 self.A += np.outer(X[i][j], X[i][j])
#         self.A += sum([np.outer(row, row) for row in X])
                
#         self.Ainv = np.linalg.inv(self.A)
        
    def update_gram_matrix(self, X):
#         print("Sherman-Morrison Formula")
        for row in X:
            self.Ainv = self.Ainv - np.dot((np.outer(np.dot(self.Ainv,row) ,row)),self.Ainv)/ (1 + np.dot(np.dot(row, self.Ainv), row))
        
        
    def run(self): # episode return
        print("UCRL-MNL")
        episode_return = []
        
        for k in tqdm(range(1,self.K+1)):
                    
            self.env.reset()
            done = 0
            R = 0
            
#             X = np.zeros((self.env.nState, self.d1*self.d2))[np.newaxis, ...]
#             Y = np.zeros(self.env.nState)[np.newaxis, ...]
#             print("===============================")
#             print("new episode")
            while not done:
                X = []
                Y = []
                
                s = self.env.state
                h = self.env.timestep
                a = self.act(s,h)
                r, s_, done = self.env.advance(a)
                
                # For debugging
                if a != 1:
                    print("k:", k, "h:", h, "a:", a, "r:", r)
                
                R += r

                ############################################################################################################
                ###### Revision                 
                # target response 
                y = np.zeros(len(self.reachable_states[s,a]))
                
                # appending input features
                for i in range(len(self.reachable_states[s,a])):
                    if list(self.reachable_states[s,a])[i] == s_:
                        y[i] = 1
                    X.append(self.varphi[(s,a,list(self.reachable_states[s,a])[i])])
                               
                self.Y.append(y)
                self.X.append(np.array(X))
                # print(self.X)
                self.update_gram_matrix(X)

                    
        
                ############################################################################################################
            
            episode_return.append(R)
            

            ############################################################################################################
#             idx = np.random.choice(range(1, len(self.X)), min(4000, len(self.X[1:])))
#             self.update_theta(k, self.X[idx], self.Y[idx])
#             if k < 30:
#                 self.update_theta(k, self.X, self.Y) # use every sample
            if k < 10:
                self.update_theta(k, self.X, self.Y) # use every sample
            ############################################################################################################
            
            self.compute_Q(k)
            
        return episode_return        
        
class RegularizedMNLRegression:

    def compute_prob(self, theta, x):
        means = np.dot(x, theta)
        u = np.exp(means)
#         u_ones = np.column_stack((u,np.ones(u.shape[0])))
#         logSumExp = u_ones.sum(axis=1)
#         prob = u_ones/logSumExp[:,None]
        logSumExp = u.sum(axis=1)
        prob = u/logSumExp[:,None]
        return prob

    def cost_function(self, theta, x, y, lam):
        m = x.shape[0]
        prob = self.compute_prob(theta, x)
        return -(1/m)*np.sum( np.multiply(y, np.log(prob))) + (1/m)*lam*np.linalg.norm(theta)

    def gradient(self, theta, x, y, lam):
        m = x.shape[0]
        prob = self.compute_prob(theta, x)
#         eps = (prob-y)[:,:-1]
        eps = (prob-y)
        grad = (1/m)*np.tensordot(eps,x,axes=([1,0],[1,0])) + (1/m)*lam*theta
        return grad

    def fit(self, x, y, theta, lam):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, messages=0, args=(x, y, lam))
        self.w = opt_weights[0]
        return self
    
class RegularizedMNLRegression_sep:

    def compute_prob(self, theta, x):
        probs = []
        for i in range(len(x)):
            means = np.dot(x[i], theta)
            u = np.exp(means)
            logSumExp = u.sum()
            prob = u/logSumExp
            probs.append(prob)
        # print(len(x), len(probs))
        return probs

    def cost_function(self, theta, x, y, lam):
        probs = self.compute_prob(theta, x)
        res = 0
        for i in range(len(x)):
            res += np.sum(np.multiply(y[i], np.log(probs[i])))
            
        m = len(x)
        res *= -(1/m)
        res += (1/m)*lam*np.linalg.norm(theta)
        
        return res

    def gradient(self, theta, x, y, lam):
        m = len(x)
        prob = self.compute_prob(theta, x)
        # print(prob)
#         print(type(prob))
#         print(y)
#         print(type(y))
        eps = [prob[i] - y[i] for i in range(len(prob))]
#         eps = (prob-y)
        
        grad = np.zeros(len(theta))
        # print(grad)
        for i in range(len(eps)):
            for j in range(len(eps[i])):
                tmp = [eps[i][j], x[i][j], eps[i][j] * x[i][j]]
                grad += eps[i][j] * x[i][j]  # GRADIENT 다 합쳐, 비용함수의 그래디언트
        grad = (1/m)*grad
        # print(grad)
        # grad += (1/m)*lam*theta        
#         grad = (1/m)*np.tensordot(eps,x,axes=([1,0],[1,0])) + (1/m)*lam*theta
        return grad

    def fit(self, x, y, theta, lam):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, messages=0, args=(x, y, lam))
        self.w = opt_weights[0]
        return self
    

env = make_riverSwim(epLen=20, nState=6)
agent = UCRL_MNL(env, K=10, c=1e-1)
agent.run()
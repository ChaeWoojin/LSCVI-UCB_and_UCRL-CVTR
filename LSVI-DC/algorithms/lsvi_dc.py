import numpy as np 
from tqdm import tqdm 

class LSVI_DC(object):
    def __init__(self, env, T, H, delta, gamma, lam):
        self.env = env
        self.T = T 
        self.H = H
        
        # Feature dimensions
        self.d1 = self.env.nState * self.env.nAction
        self.d2 = self.env.nState
        self.d = self.d1 * self.d2

        # Gram matrix
        self.A = np.identity(self.d1 * self.d2)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros(self.d)

        self.gamma = gamma
        self.lam = lam  # 1/B^2
        self.B = np.sqrt(1 / lam)
        self.delta = delta

        # Reachable states
        self.reachable_states = {(s, a): {} for s in self.env.states.keys() for a in range(self.env.nAction)}
        for s in self.env.states.keys():
            self.reachable_states[(s, 0)] = {max(s - 1, 0)}
            self.reachable_states[(s, 1)] = {s, min(s + 1, self.env.nState - 1), max(0, s - 1)}
        
        # Support state
        self.support_states = {(s, a): list(self.reachable_states[(s, a)])[0] for s in self.env.states.keys() for a in range(self.env.nAction)}
        
        # Initialize features
        self.varphi = self._initialize_varphi()  # Reward Feature Map
        self.phi = self._initialize_phi()        # Transition Feature Map

        # Initialize theta
        self.theta = np.zeros(self.d)

    def _initialize_varphi(self):
        varphi = {(s, a): np.zeros(self.d) for s in self.env.states.keys() for a in range(self.env.nAction)}
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                for s_ in range(self.env.nState):
                    r_true = self.env.R[(s, a)][0]  # Assuming reward is stored as (mean, std)
                    idx = s * self.env.nState * self.env.nAction + a * self.env.nState + s_
                    varphi[(s, a)][idx] = r_true  # Set varphi(s, a) such that <varphi(s, a), theta> = sum_{s_} P(s_|s,a)r(s, a) = r(s, a)
        return varphi

    def _initialize_phi(self):
        # One-hot encoding for transitions: phi(s, a, s') -> size (nState * nAction * nState)
        phi = {(s, a, s_): np.zeros(self.env.nState * self.env.nAction * self.env.nState) for s in range(self.env.nState) for a in range(self.env.nAction) for s_ in range(self.env.nState)}
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                for s_ in range(self.env.nState):
                    idx = s * self.env.nState * self.env.nAction + a * self.env.nState + s_
                    phi[(s, a, s_)][idx] = 1  # One-hot encoding for transitions
        return phi

    def mixture(self, s, a, u):
        return np.sum(np.array([np.multiply(u[s_], self.phi[(s, a, s_)]) for s_ in range(self.env.nState)]), axis=0)

    def act(self, s, Q_xi_1):
        return self.env.argmax(Q_xi_1)

    def Beta(self, t):
        return self.H * np.sqrt(self.d * np.log((self.lam + ((self.H ** 2) * t)) / (self.delta * self.lam))) + np.sqrt(self.lam) * self.B

    def OVI(self, N): # Optimistic value iteration
        # Initialize
        Q = np.zeros((N + 1, self.env.nState, self.env.nAction))
        V = np.zeros((N + 1, self.env.nState))
        W = np.zeros((N + 1, self.env.nState))
        Q[0, :, :] = np.ones((self.env.nState, self.env.nAction)) / (1 - self.gamma)
        V[0, :] = np.ones(self.env.nState) / (1 - self.gamma)
        
        # Update confidence set constraint
        t_k = self.env.timestep
        theta_k = self.theta.copy()
        Ainv_k = np.linalg.inv(self.Ainv).copy() 
        beta_k = self.Beta(t_k).copy()

        for n in tqdm(range(N), desc="Value Iteration (LSVI-DC)", leave=False):
            W[n, :] = V[n, :] - np.min(V[n, :])
            
            # Linear Mixture MDP Setting
            for s in range(self.env.nState):
                for a in range(self.env.nAction):
                    vphi = self.varphi[s, a] + self.gamma * self.mixture(s, a, W[n, :])
                    Q[n + 1, s, a] = min(np.dot(vphi, theta_k) + self.gamma * np.min(V[n, :]) + beta_k * np.sqrt(np.dot(vphi.T, np.dot(Ainv_k, vphi))), 
                                         1 / (1 - self.gamma))
                V[n + 1, s] = np.max(Q[n + 1, s, :])
                V[n + 1, s] = min(V[n + 1, s], np.min(V[n + 1, :]) + self.H)
            
        
        print(Q[N, :, :])
        return Q, V, W
        

    def run(self):
        print('LSVI_DC')
        episode_return = []  # round_return

        t_k = 1
        N_k = self.T - t_k + 1
        Q, V, W = self.OVI(N=N_k)
        A_k = self.A.copy()  # Copy of the Gram matrix
        R = 0
        for t in tqdm(range(1, self.T + 1)):
            if np.linalg.det(self.A) > 2 * np.linalg.det(A_k):  # Update at episode boundaries
                t_k = t
                A_k = self.A.copy()
                N_k = self.T - t_k + 1
                Q, V, W = self.OVI(N_k) # Perform EVI
                            
            st = self.env.state
            xi = np.array([np.argmax(Q[self.T - t + 1 : N_k + 1, st, a]) + self.T - t + 1 
                          for a in range(self.env.nAction)])
            Qt = np.array([Q[xi[a], st, a] for a in range(self.env.nAction)])
            # print(st, Qt)
            at = self.act(st, Qt)

            r, s_ = self.env.advance(at)
            R += r

            Wt = W[xi[at]-1, :]
            vphit = self.varphi[st, at]
            phit = self.mixture(st, at, Wt)
            tmp = vphit + self.gamma * phit
            
            self.A += np.outer(tmp, tmp)
            self.Ainv -= np.dot(np.outer(np.dot(self.Ainv, tmp), tmp), self.Ainv) / (1 + np.dot(np.dot(tmp, self.Ainv), tmp))
            self.b += np.multiply(tmp, Wt[s_])

            self.theta = np.dot(self.Ainv, self.b)

            episode_return.append(R)

        return episode_return
    
import sys
sys.path.append("../env/")
from env import *

env = make_riverSwim(T=10000, nState=6)
agent = LSVI_DC(env, T=10000, H=1, delta=0.01, gamma=0.99, lam=10)    
print(agent.run())

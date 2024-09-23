import numpy as np
from tqdm import tqdm
# import cvxpy as cp

# import sys
# sys.path.append(r'/Users/optalab/Documents/UCRL-MNL/')
# from env_hard import *

class UCRL2:
    def __init__(self, env, T, delta):
        self.env = env
        self.T = T
        self.delta = delta
        # self.epsilon = epsilon
        
        # Initialize counts and estimates
        self.N_sa = {}
        self.N_sas = {}
        self.R_sa_hat = {}
        self.r_sa_hat = {}
        self.P_sa_hat = {}
        self.v_k_sa = {}

        self.d_p = {} 
        self.d_r = {}
        for s in self.env.states.keys():
                self.N_sa[s] = np.zeros(self.env.nAction)
                self.v_k_sa[s] = np.zeros(self.env.nAction)
                for a in range(self.env.nAction):
                    self.N_sas[s, a] = np.zeros(self.env.nState)
                    self.R_sa_hat[s, a] = [0, 0]
                    self.r_sa_hat[s, a] = [0, 0]
                    self.d_p[s,a] = 0
                    self.d_r[s,a] = 0
                    
        
        # # reachable states
        self.reachable_states = {(s,a):{} for s in self.env.states.keys() for a in range(self.env.nAction)}

        for a in range(self.env.nAction):  # hard 기준
            self.reachable_states[(0,a)] = {0, 1}
            self.reachable_states[(1,a)] = {0, 1}
        
        # support state
        # self.support_states = {(s,a):list(self.reachable_states[(s,a)])[0] for s in self.env.states.keys() for a in range(self.env.nAction)}
        
        for s in self.env.states.keys():
            for a in range(self.env.nAction):
                self.P_sa_hat[s,a] = np.zeros(self.env.nState)
                for s_ in self.reachable_states[(s,a)]:
                    self.P_sa_hat[s,a][s_] = 1 / len(self.reachable_states[(s,a)])
                    

        
    def confidence_bounds(self):
        for s in self.env.states.keys():
            for a in range(self.env.nAction):
                self.d_p[s,a] = np.sqrt(14 * self.env.nState * np.log(2 * self.env.nState * self.env.nAction / self.delta) / max(1, self.N_sa[s][a]))
                self.d_r[s,a] = np.sqrt(7 * np.log(2 * self.env.nState * self.env.nAction / self.delta) / (2 * max(1, self.N_sa[s][a])))
        
    
    def EVI(self, t_k):
        # epsilon = min(0.01, 1 / np.sqrt(t_k))
        epsilon = 1 / np.sqrt(t_k)
        cnt = 0
        u = {s: 0.0 for s in self.env.states.keys()}
        
        while True:
            cnt += 1
            u_old = u.copy()
            # u_sort = sorted(u.items(), key=lambda x: x[1], reverse=True)

            p = self.P_sa_hat.copy()
            
            # for s in self.env.states.keys():
            #     for a in range(self.env.nAction):
            #         p[s,a] = np.zeros(self.env.nState)
            #         for s_ in self.reachable_states[(s,a)]:
            #             self.P_sa_hat[s,a][s_]

            for s in self.env.states.keys():
                Q = np.zeros(self.env.nAction)
                for a in range(self.env.nAction):
                    r_tilde_sa = self.r_sa_hat[s, a][0] + self.d_r[s,a]
                    idx_max = np.argmax(p[s, a])
                    p[s, a][idx_max] = np.minimum(1.0, self.P_sa_hat[s, a][idx_max] + self.d_p[s,a] * 0.5)
                    p[s, a] /= p[s, a].sum()

                    Q[a] = r_tilde_sa + sum( [p[s, a][s_] * u[s_] for s_ in self.reachable_states[(s,a)] ] )
                u[s] = np.max(Q)

            # Check convergence
            # diff = np.max([u[s] - u_old[s] for s in self.env.states.keys()])
            # diff_1 = max(abs(u[s] - u_old[s]) for s in self.env.states.keys())
            # diff_2 = min(abs(u[s] - u_old[s]) for s in self.env.states.keys())
            diff = max(abs(u[s] - u_old[s]) for s in self.env.states.keys()) - min(abs(u[s] - u_old[s]) for s in self.env.states.keys())
            if cnt == 1e2 or  diff <= epsilon:
                print(cnt)
                break

        # Extract optimal policy
        pi = {s: 0.0 for s in self.env.states.keys()}
        for s in self.env.states.keys():
                Q = np.zeros(self.env.nAction)
                for a in range(self.env.nAction):
                    r_tilde_sa = self.r_sa_hat[s, a][0] + self.d_r[s,a]
                    idx_max = np.argmax(p[s, a])
                    p[s, a][idx_max] = np.minimum(1.0, self.P_sa_hat[s, a][idx_max] + self.d_p[s,a] * 0.5)
                    p[s, a] /= p[s, a].sum()

                    Q[a] = r_tilde_sa + sum( [p[s, a][s_] * u[s_] for s_ in self.reachable_states[(s,a)] ] )
                pi[s] = np.argmax(Q)  # s에서 Q 제일 크게 하는액션
        return u, pi
               
    
    def act(self, s, pi_tilde):
        tmp = {}
        for key, value in pi_tilde.items():
            if s in key:
                tmp[key] = value

        return self.env.argmax(np.array(tmp))

        
    def run(self):
        print('UCRL2')
        episode_return = []
        R = 0
        t_k = 1
        
        for t in tqdm(range(1, self.T+1)):
            
            s = self.env.state
            if t_k == 1:
                a = np.random.choice([a for a in range(self.env.nAction)])  # line 3, 초기 둘중 하나 선택
            else:
                # a = self.act(s, pi_tilde)
                # a = self.act(s, u)
                a = pi_tilde[s]
            
            r, s_ = self.env.advance(a)
            R += r
            
            self.N_sa[s][a] += 1
            self.v_k_sa[s][a] += 1  #   self.v_k_sa[s] = np.zeros(self.env.nAction)


            self.N_sas[s, a][s_] += 1
            self.R_sa_hat[s, a][0] += r
            
             # episode update
            if self.v_k_sa[s_][a] >= max(1, self.N_sa[s_][a]): 
                # 1.
                t_k = t+1
                # 2.
                for s in self.env.states.keys():
                    self.v_k_sa[s] = np.zeros(self.env.nAction)

                for s in self.env.states.keys():
                    for a in range(self.env.nAction):
                        self.r_sa_hat[s, a][0] = self.R_sa_hat[s, a][0] / max(1, self.N_sa[s][a])
                        for s_next in self.reachable_states[(s,a)]:
                            self.P_sa_hat[s, a][s_next] = self.N_sas[s, a][s_next] / max(1, self.N_sa[s][a])
                
                self.confidence_bounds()
                u, pi_tilde = self.EVI(t_k)
                # pi_tilde = self.EVI(s, t_k)


            episode_return.append(R)
    
        return episode_return

# # # Usage:
# env = make_hardToLearnMDP(T=100)
# agent = UCRL2(env, T=100, delta=0.1)
# returns = agent.run()
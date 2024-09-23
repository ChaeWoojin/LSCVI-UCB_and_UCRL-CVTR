import numpy as np
from tqdm import tqdm
# import cvxpy as cp

# import sys
# sys.path.append(r'/Users/optalab/Documents/UCRL-MNL/')
# from env import *

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

        
        for s in self.env.states.keys():  # river_swim 기준
            self.reachable_states[(s,0)] = {max(s-1, 0)}
            self.reachable_states[(s,1)] = {s, min(s+1, self.env.nState-1), max(0, s-1)}
        
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
                
    # 현재 state에서 u를 최대 크게하는 다음 state로 갈 수 있는 확률이 가장 높은 action 선택.
    def act(self, u, s, pi_tilde):
        tmp = {}
        lst = []
        max_value = max(u.values())

        for key, value in u.items():
            if value == max_value:
                lst.append(key)

        s_next = np.random.choice(lst)

        for key, value in pi_tilde.items():
            if s == key[0]:
                tmp[key] = value
        
        next_action = []
        max_prob = max(tmp[s, a][s_next]  for a in range(self.env.nAction))
        for a in range(self.env.nAction):
            if tmp[s, a][s_next] == max_prob:
                next_action.append(a)

        return np.random.choice(next_action)

        
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
                # a = self.act(u, s, pi_tilde)
                # a = self.act(s, u)
                a = pi_tilde[s]
            
            r, s_ = self.env.advance(a)
            R += r
            if s == 5:
                print(f't: {t}, R: {R}, s: {s}, a: {a}, r: {r}')

            self.N_sa[s][a] += 1
            self.v_k_sa[s][a] += 1  #   self.v_k_sa[s] = np.zeros(self.env.nAction)


            self.N_sas[s, a][s_] += 1
            self.R_sa_hat[s, a][0] += r
            
            # episode update
            if self.v_k_sa[s_][a] >= max(1, self.N_sa[s_][a]) and t > 20: 
                # 1.
                t_k = t+1
                # print(t_k)
                # 2.
                for s in self.env.states.keys():
                    self.v_k_sa[s] = np.zeros(self.env.nAction)

                for s in self.env.states.keys():
                    for a in range(self.env.nAction):
                        self.r_sa_hat[s, a][0] = self.R_sa_hat[s, a][0] / max(1, self.N_sa[s][a])
                        if np.sum(self.N_sas[s, a]) == 0:
                            continue
                        for s_next in self.reachable_states[(s,a)]:                            
                            self.P_sa_hat[s, a][s_next] = self.N_sas[s, a][s_next] / max(1, self.N_sa[s][a])
                

                self.confidence_bounds()
                u, pi_tilde = self.EVI(t_k)
                # pi_tilde = self.EVI(s, t_k)


            episode_return.append(R)
    
        return episode_return

# # Usage:
# env = make_riverSwim(T=500, nState=6)
# agent = UCRL2(env, T=500, delta=0.1)
# returns = agent.run()
import numpy as np
from tqdm import tqdm

class UCRL_VTR(object):
    '''
    Algorithm 1 as described in the paper Model-Based RL with
    Value-Target Regression
    The algorithm assumes that the rewards are in the [0,1] interval.
    '''
    def __init__(self,env,K,c):
        self.env = env
        self.K = K
        # A unit test that randomly explores for a period of time then learns from that experience
        # Here self.random_explore is a way to select a period of random exploration.
        # When the current episode k > total number of episodes K divided by self.random_explore
        # the algorithm switches to the greedy action with respect to its action value Q(s,a).
        # Here the dimension (self.d) for the Tabular setting is |S x A x S| as stated in Appendix B
        self.d = self.env.nState * self.env.nAction * self.env.nState
        # In the tabular setting the basis models is just the dxd identity matrix, see Appendix B
        self.P_basis = np.identity(self.d)
        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(env.epLen + 1)} # self.V[env.epLen] stays zero
        #self.create_value_functions()
        #The index of each (s,a,s') tuple, see Appendix B
        self.sigma = {}
        self.state_idx = {}
        self.createIdx()
        #See Step 2, of algorithm 1
#         self.M = env.epLen**2*self.d*np.identity(self.d)
        # For use in the confidence bound bonus term, see Beta function down below
        self.lam = 1.0
        #Self.L is no longer need, but will keep for now.
        self.L = 1.0
        self.M = np.identity(self.d)*self.lam
        self.Minv = np.identity(self.d)*(1/self.lam)
        #See Step 2
        self.w = np.zeros(self.d)
        #See Step 2
        self.theta = np.dot(self.Minv,self.w)
        #See Step 3
        self.delta = 1./self.K
        #m_2 >= the 2-norm of theta_star, see Bandit Algorithms Theorem 20.5
        #self.error()
        #self.m_2 = np.linalg.norm(self.true_p) + 0.1
        self.m_2 = np.sqrt(self.env.nState*self.env.nAction)
        self.d1 = env.nState * env.nAction
        self.c = c




    def feature_vector(self,s,a,h):
        '''
        Returning sum_{s'} V[h+1][s'] P_dot(s'|s,a),
        with V stored in self.
        Inputs:
            s - the state
            a - the action
            h - the current timestep within the episode
        '''
        sums = np.zeros(self.d)
        for s_ in self.env.states.keys():
            #print(s,s_)
            sums += self.V[h+1,s_] * self.P_basis[self.sigma[(s,a,s_)]]
        return sums

    def proj(self, x, lo, hi):
        '''Projects the value of x into the [lo,hi] interval'''
        return max(min(x,hi),lo)

    def update_Q(self,s,a,k,h):
        '''
        A function that updates both Q and V, Q is updated according to equation 4 and
        V is updated according to equation 2
        Inputs:
            s - the state
            a - the action
            k - the current episode
            h - the current timestep within the episode
        Currently, does not properly compute the Q-values but it does seem to learn theta_star
        '''
        #Here env.R[(s,a)][0] is the true reward from the environment
        # Alex's code: X = self.X[h,:]
        # Suggested code:
        X = self.feature_vector(s,a,h)
        self.Q[h,s,a] = self.proj(self.env.R[(s,a)][0] + np.dot(X,self.theta) + self.Beta(h) \
            * np.sqrt(np.dot(np.dot(np.transpose(X),self.Minv),X)), 0, self.env.epLen)
        self.V[h,s] = max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

    def update_Qend(self,k):
        '''
        A function that updates both Q and V at the end of each episode, see step 16 of algorithm 1
        Inputs:
            k - the current episode
        '''
        #step 16
        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    #Here env.R[(s,a)][0] is the true reward from the environment
                    # Alex's code: X = self.X[h,:]
                    # Suggested code:
                    self.update_Q(s,a,k,h)
                self.V[h,s] = max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

    def update_stat(self,s,a,s_,h):
        '''
        A function that performs steps 9-13 of algorithm 1
        Inputs:
            s - the current state
            a - the action
            s_ - the next state
            k - the current episode
            h - the timestep within episode when s was visited (starting at zero)
        '''
        #Step 10
#         self.X[h,:] = self.feature_vector(s,a,h) # do not need to store this
        X = self.feature_vector(s,a,h)
        #Step 11
        y = self.V[h+1,s_]
#         if s_ != None:
#             y = self.V[h+1][s_]
#         else:
#             y = 0.0
        #Step 12
        self.M = self.M + np.outer(X,X)
        self.Minv = self.Minv - np.dot((np.outer(np.dot(self.Minv,X),X)),self.Minv) / \
                    (1 + np.dot(np.dot(X,self.Minv),X))
        #Step 13
        self.w = self.w + y*X

    def update_param(self):
        '''
        Updates our approximation of theta_star at the end of each episode, see
        Step 15 of algorithm1
        '''
        #Step 15
        self.theta = np.matmul(self.Minv,self.w)

    def act(self,s,h,k):
        '''
        Returns the greedy action with respect to Q_{h,k}(s,a) for a \in A
        see step 8 of algorithm 1
        Inputs:
            s - the current state
            h - the current timestep within the episode
        '''
        
        return self.env.argmax(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
        

    def createIdx(self):
        '''
        A simple function that creates sigma according to Appendix B.
        Here sigma is a dictionary who inputs is a tuple (s,a,s') and stores
        the interger index to be used in our basis model P.
        '''
        i = 0
        j = 0
        k = 0
        for s in self.env.states.keys():
            self.state_idx[s] = int(j)
            j += 1
            for a in range(self.env.nAction):
                for s_ in self.env.states.keys():
                    self.sigma[(s,a,s_)] = int(i)
                    i += 1

    def Beta(self,h):
        '''
        A function that return Beta_k according to Algorithm 1, step 3
        '''
        #Step 3
        #Confidence bound from Appendix F/Chapter 20 of Bandit Algorithms Chpt 20 (Lattimore/Szepesvari).
        first = np.sqrt(self.lam)*self.m_2
        (sign, logdet) = np.linalg.slogdet(self.M)
        det = sign * logdet
        second = (self.env.epLen-h)/2*np.sqrt(2*np.log(1/self.delta) + min(det,pow(10,10)) - np.log(pow(self.lam,self.d)))
        
        return first + self.c*second
    
    def getweightedL1(self):
        self.weights = self.count
        self.true_p = np.zeros((self.env.nState,self.env.nAction,self.env.nState))
        for s in range(self.env.nState):
            for a in range(self.env.nAction):
                for s_ in range(self.env.nState):
                    self.true_p[s,a,s_] = self.env.P[s,a][s_]
                    
                    #for numerical stability
                    if sum(self.count[s,a,:]) == 0:
                        self.weights[s,a,s_] = self.count[s,a,s_]/1.0
                        
                    else:
                        self.weights[s,a,s_] = self.count[s,a,s_]/sum(self.count[s,a,:])
                        
        self.weights = self.weights.reshape(self.env.nState*self.env.nAction*self.env.nState)
        self.true_p = self.true_p.reshape(self.env.nState*self.env.nAction*self.env.nState)
        temp = 0
        for i in range(self.env.nState*self.env.nAction*self.env.nState):
            temp += abs(self.theta[i]-self.true_p[i])*self.weights[i]
        return temp

    def run(self):
        '''
        Simulates the agent interacting with the environment over K episodes
        Input: Nothing
        Output: A Kx1 reward vector and a Kx1 model error vector
        '''
        E_return = []
        
        #Stores counts for use in weighted L1 norm
        self.count = np.zeros((self.env.nState,self.env.nAction,self.env.nState))
        self.model_error = np.zeros(self.K)
        print(self.name())
        
        for k in tqdm(range(1,self.K+1)):
            self.env.reset()
            done = 0
            R = 0
            while done != 1:
                s = self.env.state
                h = self.env.timestep
                a = self.act(s,h,k)
                r,s_,done = self.env.advance(a)
                self.count[s,a,s_] += 1
                
                R += r
                
                self.update_stat(s,a,s_,h)
            self.update_param()
            self.update_Qend(k)
            E_return.append(R)
            self.model_error[k-1] = self.getweightedL1()
        return E_return

    def name(self):
        return 'UCRL_VTR'



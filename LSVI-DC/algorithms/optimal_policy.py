
class Optimal_Policy(object):
    
    def __init__(self, env, T):
        self.T = T
        self.env = env
        
    def run(self):
        print("Go to the Right!")
        episode_return = []
        state_return = []
        
        R = 0
        for t in range(1,self.T+1):
            # self.env.reset()
            # print(t)
            # while not done:
                # s = self.env.state
                # h = self.env.timestep
            s = self.env.state
            a = 1
            r, s_ = self.env.advance(a)                
            R += r            
            episode_return.append(R)
            state_return.append(s)
            
        return episode_return, state_return
        

class Optimal_Policy(object):
    
    def __init__(self, env, T):
        self.T = T
        self.env = env
        
    def run(self):
        print("Go to the state ONE!!!")
        episode_return = []
        
        R = 0
        for t in range(1,self.T+1):
            # self.env.reset()
            # print(t)
            # while not done:
                # s = self.env.state
                # h = self.env.timestep
            s = self.env.state
            # tmp = max(self.env.P[s, i][1] for i in range(self.env.nAction))
            # for i in range(self.env.nAction):
            #     if self.env.P[s, i][1] == tmp:
            #         a = i
            #         break
            a = 1
            r, s_ = self.env.advance(a)                
            R += r           
            # print(f't: {t}, R: {R}, s: {s}, a: {a}')
            episode_return.append(R)
            
        return episode_return
        
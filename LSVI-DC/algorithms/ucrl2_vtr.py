import numpy as np
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB

class UCRL2_VTR(object):
    def __init__(self, env, T, c, delta, lam, epsilon):
        self.env = env
        self.T = T
        
        # Feature dimensions
        self.d1 = self.env.nState * self.env.nAction
        self.d2 = self.env.nState
        self.d = self.d1 * self.d2

        # Gram matrix
        self.A = np.identity(self.d1 * self.d2)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros(self.d)

        self.lam = lam  # 1/B^2
        self.B = np.sqrt(1 / lam)
        self.delta = delta
        self.epsilon = epsilon

        # Reachable states
        self.reachable_states = {(s, a): {} for s in self.env.states.keys() for a in range(self.env.nAction)}
        for s in self.env.states.keys():
            self.reachable_states[(s, 0)] = {max(s - 1, 0)}
            self.reachable_states[(s, 1)] = {s, min(s + 1, self.env.nState - 1), max(0, s - 1)}
        
        # Support state
        self.support_states = {(s, a): list(self.reachable_states[(s, a)])[0] for s in self.env.states.keys() for a in range(self.env.nAction)}
        
        # Initialize features
        self.phi = self._initialize_phi()
        self.psi = self._initialize_psi()
        self.varphi = self._initialize_varphi()
        # print(self.varphi)

        # Initialize theta
        self.theta = np.zeros(self.d)

        # Confidence radius param
        self.c = c

        # Initialize Gurobi model once
        self.gurobi_model, self.theta_var = self.init_gurobi_model()

    def init_gurobi_model(self):
        # Create the Gurobi model only once
        model = gp.Model("solve_max_theta")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Define the optimization variable for theta
        theta_var = model.addMVar(self.d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")

        # Add non-negativity constraints for transition probabilities
        for s in self.psi.keys():
            for a in range(self.env.nAction):
                for s_ in self.reachable_states[(s, a)]:
                    varphi = self.varphi[(s, a, s_)]
                    model.addConstr(varphi @ theta_var >= 0, name=f"probability_nonneg_{s}_{a}_{s_}")

        # Add constraint that the sum of transition probabilities must equal 1 for each (s, a)
        for s in self.psi.keys():
            for a in range(self.env.nAction):
                varphi_sum = gp.quicksum(self.varphi[(s, a, s_)] @ theta_var for s_ in self.reachable_states[(s, a)])
                model.addConstr(varphi_sum == 1, name=f"probability_sum_{s}_{a}")

        return model, theta_var

    def _initialize_phi(self):
        phi = {(s, a): np.zeros(self.d1) for s in self.env.states.keys() for a in range(self.env.nAction)}
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
        varphi = {(s, a, s_): np.zeros(self.d) for s in self.psi.keys() for a in range(self.env.nAction) for s_ in self.reachable_states[s, a]}
        for s in self.psi.keys():
            for a in range(self.env.nAction):
                for s_ in self.reachable_states[(s, a)]:
                    # varphi[(s, a, s_)] = np.outer(self.phi[(s, a)], 
                    #                               self.psi[s_]  - self.psi[self.support_states[s,a]]).flatten()
                    varphi[(s, a, s_)] = np.outer(self.phi[(s, a)], self.psi[s_]).flatten()
        return varphi

    def mixture(self, s, a, u):
        return np.sum(np.array([np.multiply(u[s_], self.varphi[(s, a, s_)]) for s_ in self.reachable_states[(s, a)]]), axis=0)

    def act(self, s, w_k):
        return self.env.argmax( np.array([self.env.R[s,a][0] + np.dot(self.theta, self.mixture(s, a, w_k)) for a in range(self.env.nAction)]) )

    def Beta(self, t_k):
        return self.env.nState * np.sqrt(self.d * np.log((self.lam + ((self.env.nState ** 2) * t_k)) / (self.delta * self.lam))) + np.sqrt(self.lam) * self.B

    def EVI(self, t_k):
        cnt = 0
        u = {s: 0.0 for s in self.env.states.keys()}
        while True:
            u_old = u.copy()
            cnt += 1
            for s in self.env.states.keys():
                max_value = -1e9
                Beta_t = self.Beta(t_k)
                for a in range(self.env.nAction):
                    phi_u = self.mixture(s, a, u)

                    # Update objective in Gurobi model
                    self.gurobi_model.setObjective(phi_u @ self.theta_var, GRB.MAXIMIZE)

                    # Solve the optimization problem
                    self.gurobi_model.optimize()
                    
                    if self.gurobi_model.status == GRB.OPTIMAL:
                        theta_sol = self.theta_var.X  # Get the optimized theta
                    else:
                        raise ValueError("Optimization did not converge")

                    # Calculate the Q-value
                    value = self.env.R[s, a][0] + np.dot(phi_u, theta_sol)
                    max_value = max(max_value, value)

                u[s] = max_value

            # Check for convergence
            if cnt == 100 or max(u[s] - u_old[s] for s in self.env.states.keys()) - min(u[s] - u_old[s] for s in self.env.states.keys()) <= self.epsilon:
                break
        return u

    def update_confidence_set_constraint(self, theta_k, beta_k, Sigma_k):
        '''
        Update the quadratic confidence set constraint: ||Sigma_k^{1/2}(theta - theta_k)||_2 <= beta_k.
        This is done once per episode, so the constraints do not need to be modified repeatedly.
        '''
        # Confidence set constraint: ||Sigma_k^{1/2}(theta - theta_k)||_2 <= beta_k
        diff_theta = self.theta_var - theta_k

        # Use gp.QuadExpr to build the quadratic constraint ||Sigma_k^{1/2}(theta - theta_k)||_2^2
        Sigma_sqrt = np.linalg.cholesky(Sigma_k)  # Cholesky decomposition of Sigma_k
        quadratic_term = gp.QuadExpr()

        # Construct the quadratic form (Sigma_sqrt @ diff_theta) @ (Sigma_sqrt @ diff_theta)
        for i in range(self.d):
            for j in range(self.d):
                quadratic_term += diff_theta[i] * Sigma_sqrt[i, j] * diff_theta[j]

        # Remove the previous confidence set constraint if it exists
        for constr in self.gurobi_model.getConstrs():
            if constr.ConstrName == 'confidence_set':
                self.gurobi_model.remove(constr)
                break  # Only one constraint, break once it's found and removed

        # Add the new quadratic constraint
        self.gurobi_model.addConstr(quadratic_term <= beta_k**2, name="confidence_set")

        # Warm-start optimization with the current theta_k
        self.theta_var.start = theta_k

    def run(self):
        print('UCRL2_VTR')
        episode_return = []  # round_return

        A_k = self.A.copy()  # Copy of the Gram matrix
        t_k = 1
        w_k = {s: 1.0 for s in self.env.states.keys()}
        R = 0
        for t in tqdm(range(1, self.T + 1)):
            if np.linalg.det(self.A) > 2 * np.linalg.det(A_k):  # Update at episode boundaries
                t_k = t
                A_k = self.A.copy()

                # Update confidence set constraint
                theta_k = self.theta.copy()
                Sigma_k = np.linalg.inv(self.A).copy()  # Assuming Sigma_k is the inverse of A
                beta_k = self.Beta(t_k).copy()
                self.update_confidence_set_constraint(theta_k, beta_k, Sigma_k)

                # Perform EVI
                u_k = self.EVI(t_k)
                probs_k = self.theta_var.X
                # for s in range(self.env.nState):
                #     for a in range(self.env.nAction):
                #         for s_ in range(self.env.nState):
                #             print("P(%d | %d, %d) = %f"%(s_, s, a, probs_k[(12*s + 6*a + s_)]), flush=True)
                            
                tmp = (max(u_k) - min(u_k)) / 2
                w_k = {s: u_k[s] - tmp for s in self.env.states.keys()}

            s = self.env.state
            if t_k == 1:
                a = np.random.choice([a for a in range(self.env.nAction)])  # Initial random action selection
            else:
                a = self.act(s, w_k)

            r, s_ = self.env.advance(a)
            R += r

            tmp = self.mixture(s, a, w_k)

            self.A += np.outer(tmp, tmp)
            self.Ainv -= np.dot(np.outer(np.dot(self.Ainv, tmp), tmp), self.Ainv) / (1 + np.dot(np.dot(tmp, self.Ainv), tmp))
            self.b += np.multiply(w_k[s_], tmp)

            self.theta = np.dot(self.Ainv, self.b)

            episode_return.append(R)

        return episode_return


# import sys
# sys.path.append("../env/")
# from env import *

# env = make_riverSwim(T=100, nState=6)
# epsilon = min(0.01, 1 / np.sqrt(100))
# agent = UCRL2_VTR(env, T=100, c=1e-2, delta=0.01, lam=1, epsilon=epsilon)    
# agent.run()

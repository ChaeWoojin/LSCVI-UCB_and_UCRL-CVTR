import numpy as np
from scipy.optimize import fmin_tnc

def calculate_pt_s_prime(s_prime, s_t, a_t, theta, phi, S_a):
    numerator = np.exp(np.dot(phi(s_t, a_t, s_prime), theta))
    denominator = np.sum([np.exp(np.dot(phi(s_t, a_t, s_double_prime), theta)) 
                          for s_double_prime in S_a])
    return numerator / denominator

def loss_function(theta, y_t_s_prime, S_t, a_t, phi, S_a):
    p_t_s_prime = np.array([calculate_pt_s_prime(s_prime, S_t, a_t, theta, phi, S_a) 
                            for s_prime in S_t])
    return -np.sum(y_t_s_prime * np.log(p_t_s_prime))

def gradient_loss(theta, y_t_s_prime, S_t, a_t, phi, S_a):
    p_t_s_prime = np.array([calculate_pt_s_prime(s_prime, S_t, a_t, theta, phi, S_a) 
                            for s_prime in S_t])
    phi_t_s_prime = np.array([phi(S_t, a_t, s_prime) for s_prime in S_t])
    return -np.sum((y_t_s_prime - p_t_s_prime)[:, np.newaxis] * phi_t_s_prime, axis=0)

def objective_function(theta, theta_hat_t, grad_l_t, sigma_hat_t, eta):
    diff = theta - theta_hat_t
    return float(np.dot(grad_l_t, diff) + (1 / (2 * eta)) * np.dot(diff, np.dot(sigma_hat_t, diff)))

def objective_gradient(theta, theta_hat_t, grad_l_t, sigma_hat_t, eta):
    diff = theta - theta_hat_t
    return grad_l_t + (1 / eta) * np.dot(sigma_hat_t, diff)

def update_theta(theta_hat_t, grad_l_t, sigma_hat_t, eta):
    obj_fn = lambda theta: objective_function(theta, theta_hat_t, grad_l_t, sigma_hat_t, eta)
    obj_grad = lambda theta: objective_gradient(theta, theta_hat_t, grad_l_t, sigma_hat_t, eta)
    
    result, _, _ = fmin_tnc(obj_fn, theta_hat_t, fprime=obj_grad, 
                            bounds=[(None, None)] * len(theta_hat_t),
                            messages=0)
    return result

def online_mirror_descent_step(t, theta_hat, S, a, y_t_s_prime, phi, eta, S_a):
    # Calculate the gradient of the loss function
    grad_l_t = gradient_loss(theta_hat[t], y_t_s_prime, S[t], a[t], phi, S_a)
    
    # Calculate Σ_t and Σ̂_t
    sigma_t = np.eye(theta_hat[t].shape[0])  # Identity matrix as placeholder
    for i in range(1, t):
        sigma_t += np.outer(gradient_loss(theta_hat[i], y_t_s_prime, S[i], a[i], phi, S_a),
                            gradient_loss(theta_hat[i], y_t_s_prime, S[i], a[i], phi, S_a))
    sigma_hat_t = sigma_t + eta * np.outer(gradient_loss(theta_hat[t], y_t_s_prime, S[t], a[t], phi, S_a),
                                           gradient_loss(theta_hat[t], y_t_s_prime, S[t], a[t], phi, S_a))
    
    # Update theta using fmin_tnc
    theta_hat_next = update_theta(theta_hat[t], grad_l_t, sigma_hat_t, eta)
    
    return theta_hat_next

# Main loop
def run_algorithm(T, S, a, phi, eta, S_a):
    theta_hat = [np.zeros(phi(S[0], a[0], S[0]).shape[0])]  # Initialize θ_1 = 0
    
    for t in range(1, T):
        y_t_s_prime = np.array([1 if S[t+1] == s_prime else 0 for s_prime in S[t]])
        theta_hat.append(online_mirror_descent_step(t, theta_hat, S, a, y_t_s_prime, phi, eta, S_a))
    
    return theta_hat

# Usage example
T = 100  # Number of time steps
S = [...]  # Your state space
a = [...]  # Your action sequence
S_a = [...]  # Your action space
phi = lambda s, a, s_prime: ...  # Your feature function
eta = 0.01  # Step size

theta_star_estimate = run_algorithm(T, S, a, phi, eta, S_a)
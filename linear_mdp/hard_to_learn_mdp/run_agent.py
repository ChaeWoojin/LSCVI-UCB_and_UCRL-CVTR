import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from environments import HardLinearMDP
from agent import LSVI_DC

np.random.seed(0)

def run_lsvi_dc(d, H, T, delta):
    '''Run LSVI-DC agent and return the total reward.'''
    # Create a new instance of the environment for LSVI-DC
    hard_mdp = HardLinearMDP(d=d, H=H, T=T)
    hard_mdp.reset()
    init_s, phi, H = hard_mdp.state, hard_mdp.phi, hard_mdp.H
    lambda_reg, B = 1, 4
    agent = LSVI_DC(hard_mdp, init_s=init_s, gamma=1-np.sqrt(np.log(T)/T), phi=phi, lambda_reg=lambda_reg, H=H, B=B, delta=delta)
    total_reward_lsvi_dc = agent.run()
    return 'LSVI_DC', total_reward_lsvi_dc

def run_optimal_policy(d, H, T):
    '''Run the optimal policy and return the total reward.'''
    # Create a new instance of the environment for the optimal policy
    hard_mdp = HardLinearMDP(d=d, H=H, T=T)
    hard_mdp.reset()
    total_reward_optimal = hard_mdp.run_optimal_policy()
    return 'Optimal_Policy', total_reward_optimal

def run_mixture_experiment():
    '''Run the LSVI-DC agent and the optimal policy concurrently and plot the results.'''
    # Parameters
    d = 8
    H = 10
    T = 1000
    delta = 0.1
    
    # Use ProcessPoolExecutor to run the agents concurrently
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_lsvi_dc, d, H, T, delta): 'LSVI_DC',
            executor.submit(run_optimal_policy, d, H, T): 'Optimal_Policy'
        }
        
        # Save the plot in the 'plot' directory
        result_dir = 'result'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)  # Create the directory if it doesn't exist

        results = {}
        for future in futures:
            agent_name, total_reward = future.result()  # Get the result from the process
            results[agent_name] = np.array(total_reward).astype(int)
            
            # Save the results to text files
            with open(f"{result_dir}/results({agent_name}).txt", "a", encoding='utf-8') as fw:
                results_str = str(results[agent_name]).replace('\n', ' ')
                fw.write(results_str)
                fw.write('\n')

    # Check if both results are available
    if 'LSVI_DC' not in results:
        print("Error: LSVI_DC did not return any results.")
        return
    if 'Optimal_Policy' not in results:
        print("Error: Optimal_Policy did not return any results.")
        return

    # Calculate regret
    regret_lsvi_dc = np.array(results['Optimal_Policy']) - np.array(results['LSVI_DC'])

    # Plot cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(results['LSVI_DC']), label='LSVI-DC (Linear MDP)')
    plt.plot(np.cumsum(results['Optimal_Policy']), label='Optimal Policy', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward vs Time')
    plt.legend()

    # Plot regret
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(regret_lsvi_dc), label='LSVI-DC')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title('Regret vs Time')
    plt.legend()

    plt.tight_layout()

    # Save the plot in the 'plot' directory
    plot_dir = 'plot'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)  # Create the directory if it doesn't exist
    plot_path = os.path.join(plot_dir, 'cumulative_rewards_and_regret.png')
    
    # Save the plot
    plt.savefig(plot_path)
    print(f"Plot saved at {plot_path}")    

    plt.show()

if __name__ == '__main__':
    run_mixture_experiment()

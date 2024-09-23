import os
import sys
sys.path.append('../')
import random
import numpy as np
from tqdm import tqdm
from env.env import *
from algorithms.ucrl2 import UCRL2
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

# Function to run a single experiment
def run_experiment(run, seed, T, nState, delta, resultDir):
    random.seed(seed)
    
    # Create the environment
    env = make_riverSwim(T=T, nState=nState)
    
    # Initialize the agent
    agent = UCRL2(env, T=T, delta=delta)
    # Run the agent
    episodic_return = agent.run()
    print("seed %d done"%(seed))

    # Save the result
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)  # Create the directory if it doesn't exist
    np.save(resultDir + f'/return{run}.npy', episodic_return)

    return episodic_return

def main():
    runs = 10  # Adjust this based on the number of parallel runs
    seeds = [123 * (i + 1) for i in range(runs)]
    T = 10000
    nState = 6
    delta = 0.01

    resultDir = '../data/riverSwim/S=' + str(nState) + ', T=' + str(T) + '/UCRL2'
    
    # Use multiprocessing to run experiments in parallel
    pool = mp.Pool(mp.cpu_count())  # Use all available CPUs

    # Use pool.starmap to distribute the runs in parallel
    results = pool.starmap(run_experiment, [(run, seeds[run], T, nState, delta, resultDir) for run in range(runs)])

    pool.close()  # Close the pool to prevent new tasks from being submitted
    pool.join()  # Wait for all worker processes to finish

    episodes = np.arange(T)

    plt.figure()
    data_mean = np.mean(results, axis=0)
    data_std = np.std(results, axis=0)
        
    plt.fill_between(episodes, data_mean + data_std, data_mean - data_std, alpha=0.2)
    plt.plot(episodes, data_mean, linewidth=1.8)
    plt.title("RiverSwim: nState=6, T=100")
    plt.xlabel("The Number of Episodes")

    plt.ylabel("Episodic Returns")
    plt.show()

    return results

if __name__ == '__main__':
    run_returns = main()
    print("All experiments completed!")

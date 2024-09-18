import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
from environments import HardLinearMixtureMDP
from agent import LSVI_DC_LIN_MIX, UCRL2_VTR

def runLsviDc(d, D, T, delta, returnDict):
    '''Run LSVI-DC agent and return the total reward.'''
    hardMdp = HardLinearMixtureMDP(d=d, D=D, T=T)
    hardMdp.reset()
    initS, phi, varphi = hardMdp.state, hardMdp.phi, hardMdp.varphi
    lambdaReg, B = 1, 4
    agent = LSVI_DC_LIN_MIX(hardMdp, init_s=initS, gamma=0.99, phi=phi, varphi=varphi, lambda_reg=lambdaReg, B=B, delta=delta)
    totalRewardLsviDc = agent.run()
    returnDict['LSVI_DC_LIN_MIX'] = totalRewardLsviDc

def runUcrl2Vtr(d, D, T, delta, returnDict):
    '''Run UCRL2-VTR agent and return the total reward.'''
    hardMdp = HardLinearMixtureMDP(d=d, D=D, T=T)
    hardMdp.reset()
    initS, phi = hardMdp.state, hardMdp.phi, hardMdp.action_rank
    lambdaReg, B = 1.2, 4
    agent = UCRL2_VTR(hardMdp, init_s=initS, gamma=0.9999, phi=phi, lambda_reg=lambdaReg, B=B, delta=delta)
    totalRewardUcrl2Vtr = agent.run()
    returnDict['UCRL2_VTR'] = totalRewardUcrl2Vtr

def runOptimalPolicy(d, D, T, returnDict):
    '''Run the optimal policy and return the total reward.'''
    hardMdp = HardLinearMixtureMDP(d=d, D=D, T=T)
    hardMdp.reset()
    totalRewardOptimal = hardMdp.run_optimal_policy()
    returnDict['Optimal_Policy'] = totalRewardOptimal

def runMixtureExperiment():
    '''Run the UCRL2, LSVI-DC agents, and the optimal policy concurrently and plot the results.'''
    # Parameters
    d = 8
    D = 5
    T = 1000
    delta = 0.2
    
    manager = Manager()
    returnDict = manager.dict()

    # Start processes
    lsviDcProcess = Process(target=runLsviDc, args=(d, D, T, delta, returnDict))
    # ucrl2VtrProcess = Process(target=runUcrl2Vtr, args=(d, D, T, delta, returnDict))
    optimalPolicyProcess = Process(target=runOptimalPolicy, args=(d, D, T, returnDict))
    
    lsviDcProcess.start()
    # ucrl2VtrProcess.start()
    optimalPolicyProcess.start()
    
    # Wait for all processes to complete
    lsviDcProcess.join()
    # ucrl2VtrProcess.join()
    optimalPolicyProcess.join()

    # Save the plot in the 'result' directory
    resultDir = 'result'
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)  # Create the directory if it doesn't exist

    results = {}
    if 'LSVI_DC_LIN_MIX' in returnDict:
        results['LSVI_DC_LIN_MIX'] = np.array(returnDict['LSVI_DC_LIN_MIX']).astype(int)
        with open(f"{resultDir}/results(LSVI_DC_LIN_MIX).txt", "a", encoding='utf-8') as fw:
            resultsStr = str(results['LSVI_DC_LIN_MIX']).replace('\n', ' ')
            fw.write(resultsStr)
            fw.write('\n')
    
    if 'Optimal_Policy' in returnDict:
        results['Optimal_Policy'] = np.array(returnDict['Optimal_Policy']).astype(int)
        with open(f"{resultDir}/results(Optimal_Policy).txt", "a", encoding='utf-8') as fw:
            resultsStr = str(results['Optimal_Policy']).replace('\n', ' ')
            fw.write(resultsStr)
            fw.write('\n')

    # Check if all results are available
    if 'LSVI_DC_LIN_MIX' not in results:
        print("Error: LSVI_DC_LIN_MIX did not return any results.")
        return
    if 'Optimal_Policy' not in results:
        print("Error: Optimal Policy did not return any results.")
        return

    # Calculate regret
    regretLsviDc = np.array(results['Optimal_Policy']) - np.array(results['LSVI_DC_LIN_MIX'])

    # Plot cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(results['LSVI_DC_LIN_MIX']), label='LSVI-DC (Linear Mixture MDP)')
    plt.plot(np.cumsum(results['Optimal_Policy']), label='Optimal Policy', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward vs Time')
    plt.legend()

    # Plot regret
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(regretLsviDc), label='LSVI-DC')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title('Regret vs Time')
    plt.legend()

    plt.tight_layout()

    # Save the plot in the 'plot' directory
    plotDir = 'plot'
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)  # Create the directory if it doesn't exist
    plotPath = os.path.join(plotDir, 'cumulative_rewards_and_regret.png')
    
    # Save the plot
    plt.savefig(plotPath)
    print(f"Plot saved at {plotPath}")    

    plt.show()

if __name__ == '__main__':
    runMixtureExperiment()

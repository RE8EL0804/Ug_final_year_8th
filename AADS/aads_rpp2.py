import csv
import bisect
import nonstationary2
from UCB import *
import aads_rpp_modules_df as aads
import joblib
import os
from pprint import pprint
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_arms(results):
    header = ['presence', 'dlt', 'dfb', 'arm', 'test_data', 'acc', 'f1',
              'tpr', 'tnr', 'prec', 'conf', 'naive']
    file_exists = os.path.isfile("../results/mab_test_output.csv")
    outfile = open("../results/mab_test_output.csv", "a")
    writer = csv.writer(outfile)
    if not file_exists:
        writer.writerow(header)
    outfile.close()
    selected_arm = aads.Model()
    test_results = []
    for i in results:
        action_set = [

            [['ip', i[1], 0.003], ['wifi', i[1], 0.003], ['zigbee', i[1], 0.003],
             ['rf', i[1], 0.003], ['audio', i[1], 0.003], ['amds'], [i[2]]],

            [['ip', i[1], 0.005], ['wifi', i[1], 0.005], ['zigbee', i[1], 0.005],
             ['rf', i[1], 0.005], ['audio', i[1], 0.005], ['amds'], [i[2]]],

            [['ip', i[1], 0.007], ['wifi', i[1], 0.007], ['zigbee', i[1], 0.007],
             ['rf', i[1], 0.007], ['audio', i[1], 0.007], ['amds'], [i[2]]],

            [['ip', i[1], 0.01], ['wifi', i[1], 0.01], ['zigbee', i[1], 0.01],
             ['rf', i[1], 0.01], ['audio', i[1], 0.01], ['amds'], [i[2]]],

            [['ip', i[1], 0.03], ['wifi', i[1], 0.03], ['zigbee', i[1], 0.03],
             ['rf', i[1], 0.03], ['audio', i[1], 0.03], ['amds'], [i[2]]],

            [['ip', i[1], 0.05], ['wifi', i[1], 0.05], ['zigbee', i[1], 0.05],
             ['rf', i[1], 0.05], ['audio', i[1], 0.05], ['amds'], [i[2]]],

            [['ip', i[1], 0.07], ['wifi', i[1], 0.07], ['zigbee', i[1], 0.07],
             ['rf', i[1], 0.07], ['audio', i[1], 0.07], ['amds'], [i[2]]],

            [['ip', i[1], 0.1], ['wifi', i[1], 0.1], ['zigbee', i[1], 0.1],
             ['rf', i[1], 0.1], ['audio', i[1], 0.1], ['amds'], [i[2]]],
        ]

        arm = action_set[i[3]]
        if i[0] == 0:
            train = "0.train"
            test = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7']#,
        else:
            train = "1.train"
            test = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7']#,

        selected_arm.train(filepath, train, arm)
        ip_mod = joblib.load(filepath + "models/" + train + ".ip.iforest.sav")
        wifi_mod = joblib.load(filepath + "models/" + train + ".wifi.iforest.sav")
        zb_mod = joblib.load(filepath + "models/" + train + ".zigbee.iforest.sav")
        rf_mod = joblib.load(filepath + "models/" + train + ".rf.iforest.sav")
        audio_mod = joblib.load(filepath + "models/" + train + ".audio.iforest.sav")
        dfb = i[1]

        for t in test:
            print("##############################")
            print("Test data: " + str(t))
            print("##############################")
            reward, ACC, F1, TPR, TNR, PREC, nc, conf = selected_arm.test(filepath, t, arm, ip_mod,
                 wifi_mod, zb_mod, rf_mod, audio_mod, risk_pref, dfb)

            performance = [ACC] + [F1] + [TPR] + [TNR] + [PREC] + [conf] + [nc]
            agg_result = i + [t] + performance
            print(agg_result)
            test_results.append(agg_result)
            try:
                outfile = open("../results/mab_test_output.csv", "a")
                writer = csv.writer(outfile)
                writer.writerow(agg_result)
                outfile.close()
                print("Result saved")

            except:
                print("CSV writer: something went wrong")
    print(test_results)

def plot_performance(learning_results, algorithm, j, k):
    averageRewardDictionaryOfAllAlgorithms = learning_results['rewards']
    averageRewards = averageRewardDictionaryOfAllAlgorithms[algorithm]

    rewards = ['Rewards']
    optimal = ['OptimalAction']
    csv_file = "../results/mab." + str(j) + "." + str(k) + ".csv"

    outfile = open(csv_file, "w")
    writer = csv.writer(outfile)
    writer.writerow(rewards)
    writer.writerows(map(lambda x: [x], averageRewards))
    outfile.close()
    print("Result saved")



def mab(feeds, train, test, filepath, risk_pref, dlt, dfb, pulls, presence):

    results = []
    for i in presence:
        if presence == 1:
            train = "1.train" 
            test_data = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7'] 
        else:
            train = "0.train"  
            test_data = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
            
        for j in dlt:
            for k in dfb:
                action_set = [

                    [['ip', j, 0.003], ['wifi', j, 0.003], ['zigbee', j, 0.003],
                     ['rf', j, 0.003], ['audio', j, 0.003], ['amds'], [k]],

                    [['ip', j, 0.005], ['wifi', j, 0.005], ['zigbee', j, 0.005],
                     ['rf', j, 0.005], ['audio', j, 0.005], ['amds'], [k]],

                    [['ip', j, 0.007], ['wifi', j, 0.007], ['zigbee', j, 0.007],
                     ['rf', j, 0.007], ['audio', j, 0.007], ['amds'], [k]],

                    [['ip', j, 0.01], ['wifi', j, 0.01], ['zigbee', j, 0.01],
                     ['rf', j, 0.01], ['audio', j, 0.01], ['amds'], [k]],
                    
                    [['ip', j, 0.03], ['wifi', j, 0.03], ['zigbee', j, 0.03],
                     ['rf', j, 0.03], ['audio', j, 0.03], ['amds'], [k]],

                    [['ip', j, 0.05], ['wifi', j, 0.05], ['zigbee', j, 0.05],
                     ['rf', j, 0.05], ['audio', j, 0.05], ['amds'], [k]],

                    [['ip', j, 0.07], ['wifi', j, 0.07], ['zigbee', j, 0.07],
                     ['rf', j, 0.07], ['audio', j, 0.07], ['amds'], [k]],

                    [['ip', j, 0.1], ['wifi', j, 0.1], ['zigbee', j, 0.1],
                     ['rf', j, 0.1], ['audio', j, 0.1], ['amds'], [k]],

                ]

                action_space = int(len(action_set))
                # N = action_space * 1000
                # N = action_space
                print('Number of RPP Arms in Bandit: ' + str(action_space))
                # bandit = Bandit()
                stationary = False
                # pulls = N*100
                # pulls = 1000
                print("Number of Pulls: " + str(pulls))
                runs = 1
                algorithm = []
                alpha = 0.1
                c = 2
                ucb = UCB(action_space, c, alpha)
                algorithm.append(ucb)

                Q, learning_results = nonstationary2.testAlgorithms(action_set, filepath, test_data,
                                        algorithm, runs, pulls, stationary, train, feeds, 'train', risk_pref, j)

                print("Q after learning: " + str(Q))
                arm_id = int(Q.index(max(Q)))
                print(Q.index(max(Q)))
                print(action_set[Q.index(max(Q))])
                presence_conf = i
                dfb_conf = j
                dlt_conf = k

                mab_result = [presence_conf] + [dfb_conf] + [dlt_conf] + [arm_id]
                results.append(mab_result)
                ### send to list for print
                plot_performance(learning_results, algorithm=ucb, j=j, k=k)

    print("Results: ")
    print(results)
    return results


if __name__ == '__main__':
 
    print('''
Amrita Anomaly Detection System

 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌
▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌          ▐░▌
▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌
▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌
 ▀         ▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀ 


''')
    while(1):
            print('###############################')
            print('''
	Please Select the Option:
	1) Train and Test the model
	2) Model Performance Metrics
	3) Quit
	''')
            print('###############################')
            opt=int(input())
            if opt==3:
                print("Thank You")
                exit()
            elif opt==2:
                print('''
        Select the performance metric: 
        1) Accuracy
        2) Rewards
	3) Precision
	4) F1-Score
	5) Back to Main Menu''')
                per_opt =int(input())
                if per_opt==1:
                    print('acc')
                elif per_opt==2:
                    print('Rew')
                elif per_opt==3:
                    print('Prec')
                elif per_opt==4:
                    print('F1')
                elif per_opt==5:
                    print("back")
                continue
            elif opt==1:
                presence = [0, 1]
                dlt = [0.3, 0.5, 0.7]
                dfb = [0.3, 0.5, 0.7]
                pulls = 100
                rp = 1
                if rp ==1:
                    risk_pref = 'TPR'
                else:
                    risk_pref = 'TNR'
                
                if presence == 1:
                    train = "1.train"
                    test = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7']
                else:
                    train = "0.train" 
                    test = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7']  
                filepath = "../data/"
                feeds = ['ip', 'wifi', 'zigbee', 'audio', 'rf', 'amds']
                results = mab(feeds, train, test, filepath, risk_pref, dlt, dfb, pulls, presence)
                print("results:", results)
                test_arms(results)

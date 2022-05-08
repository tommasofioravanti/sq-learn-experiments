from sklearn.QuantumUtility.Utility import *
warnings.simplefilter("ignore")

epsilon = 0.1

median = False
consistent_pe=False
p_e = False

if median:
    median = median_evaluation(phase_estimation, gamma=0.1, Q=5, omega=0.123, epsilon=0.01, nqubit=False,
                           plot_distribution=False,
                           success_prob='QAE')
    print('median evaluation:', median)

thetas = [0.35, 0.1]
if consistent_pe:
    print('Consistent_PE')
    for _ in range(5):
        cpe_estimations = [consistent_phase_estimation(epsilon=epsilon, delta=0.1, omega=th) for th in thetas]
        print()
        #print(cpe_estimations)
if p_e:
    for _ in range(5):
        pe_estimations = [phase_estimation(omega=th, epsilon=epsilon,delta=0.1) for th in thetas]

        print('P.E:', pe_estimations)

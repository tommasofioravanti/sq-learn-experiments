import numpy as np
import matplotlib.pyplot as plt
from sklearn.QuantumUtility.Utility import *
import warnings
from fitter import *
from scipy.stats import shapiro

warnings.filterwarnings("ignore")
# LOAD ARRAY
v_longer = create_rand_vec(1, 2000)[0]

v_exp = np.load('array784_exp.npy')
v_uni = np.load('array784.npy')
v_sparse20 = np.load('sparse_arr_20.npy')
v_sparse50 = np.load('sparse_arr_50.npy')

# CREATE NEW ARRAY
# new_vec = np.array(create_rand_vec(1, 100))

list_ = [v_exp, v_uni, v_sparse20, v_sparse50]
delta = 0.05


# MAKE TOMOGRAPHY

def decreasing_error_plot(vector_list, delta, norm='L2'):
    name = ['v_exp', 'v_uni', 'v_sparse80', 'v_sparse50']
    color = ['blue', 'orange', 'green', 'red']
    if len(vector_list) > 1:
        it = iter(vector_list)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('not all lists have same length!')
    if norm == 'L2':
        N = int((36 * len(vector_list[0]) * np.log(len(vector_list[0]))) / (delta ** 2))
    else:
        N = int((36 * np.log(len(vector_list[0]))) / (delta ** 2))
    ax_line_list=[]
    ax_point_list=[]
    point_coord=[]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for e, vector in enumerate(vector_list):
        if np.isclose(np.linalg.norm(vector), 1, rtol=1e-2):
            pass
        else:
            vector = vector / np.linalg.norm(vector, ord=2)
        dictionary_estimates = real_tomography(vector, delta=delta, stop_when_reached_accuracy=False, norm=norm)
        measure = list(dictionary_estimates.keys())
        samples = list(dictionary_estimates.values())
        if norm == 'L2':
            samples_ = [np.linalg.norm(vector - samples[i], ord=2) for i in range(len(samples))]
        else:
            samples_ = [np.linalg.norm(vector - samples[i], ord=np.inf) for i in range(len(samples))]

        idx = np.argwhere(
            np.diff(np.sign(np.full(shape=len(samples_), fill_value=delta) - samples_))).flatten()

        ax_point_list.append(ax.plot(measure[idx[0]], delta, 'ro', color=color[e]))
        point_coord.append(str((measure[idx[0]], delta)))
        # plt.annotate(str(measure[idx[0]], delta),(measure[idx[0]], delta))
        ax_line_list.append(ax.plot(measure, samples_)) # , label=name[e]+str((measure[idx[0]], delta))
        # plt.axvline(x=measure[idx[0]], ymax=0.2, color='black', linestyle='--')
    #ff=ax_line_list[0][0]
    plt.axhline(y=delta, color='k', linestyle='--', label=r'$\delta$=' + str(delta))
    plt.axvline(x=N, color='magenta', linestyle=':', label='N =' + str(N))
    plt.xlabel('number of measurements')
    plt.ylabel('error')
    plt.xscale('log')
    leg1=ax.legend([ax_line_list[0][0],ax_line_list[1][0],ax_line_list[2][0],ax_line_list[3][0]],
                   ['exponential','uniform','sparse80%','sparse50%'],loc=7)
    leg2 = ax.legend([ax_point_list[0][0], ax_point_list[1][0], ax_point_list[2][0], ax_point_list[3][0]],
                     [point_coord[0], point_coord[1], point_coord[2], point_coord[3]],title='coordinates',loc=9)
    plt.legend(loc=1)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    plt.savefig('tomo_executive.pdf')
    plt.show()


def make_real_predicted_comparison(vector, delta, norm='L2'):
    if np.linalg.norm(vector) != 0.999 or np.linalg.norm(vector) != 1.0:
        vector = vector / np.linalg.norm(vector, ord=2)

    def compute_predicted_measure(len_vec, delta, norm):
        if norm != 'all':
            if norm == 'L2':
                return int((36 * len_vec * np.log(len_vec)) / (delta ** 2))
            else:
                return int((36 * np.log(len_vec)) / (delta ** 2))
        else:
            return (int((36 * len_vec * np.log(len_vec)) / (delta ** 2)), int((36 * np.log(len_vec)) / (delta ** 2)))

    if norm != 'all':

        dictionary_estimates = real_tomography(vector, delta=delta, stop_when_reached_accuracy=False,
                                               norm=norm)
        measure = list(dictionary_estimates.keys())
        samples = list(dictionary_estimates.values())
        if norm == 'L2':
            samples_ = [np.linalg.norm(vector - samples[i], ord=2) for i in range(len(samples))]
            N = int((36 * len(vector) * np.log(len(vector))) / (delta ** 2))

        else:
            samples_ = [np.linalg.norm(vector - samples[i], ord=np.inf) for i in range(len(samples))]
            N = int((36 * np.log(len(vector))) / (delta ** 2))
        idx = np.argwhere(
            np.diff(np.sign(np.full(shape=len(samples_), fill_value=delta) - samples_))).flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        point_true, = ax.plot(measure[idx[0]], delta, 'ro', color='orange')#, label=str((measure[idx[0]], delta))
        point_theory, = ax.plot(N, delta, 'ro', color='blue')#, label=str((N, delta))
        '''plt.title("Comparison between real and predicted measurements",
                  fontdict={'family': 'serif',
                            'color': 'darkblue',
                            'weight': 'bold',
                            'size': 18})'''
        predicted_points = [compute_predicted_measure(len(vector), i, norm=norm) for i in samples_]

        plt.axhline(y=delta, color='k', linestyle='--', label=r'$\delta$=' + str(delta))
        plt.axvline(x=N, color='magenta', linestyle='dotted', label='N=' + str(N))
        plt.xscale('log')
        plt.ylabel('error')
        plt.xlabel('number of measurements')
        line_theory, = ax.plot(predicted_points, samples_)  #, label='theoretical' label="theory" + str((N, delta))
        line_true, = ax.plot(measure, samples_)  # , label='real', label="real" + str((measure[idx[0]], delta))

        leg1 = ax.legend([line_true,line_theory ], [ 'real','theoretical'], loc=2)
        leg2 = ax.legend([point_true, point_theory], [str((measure[idx[0]], delta)), str((N, delta))], title='coordinates'
                                                                                                             ,loc=9)
        plt.legend(loc=1)
        ax.add_artist(leg1)
        ax.add_artist(leg2)
        plt.savefig('real_theory.pdf')
        plt.show()

    else:
        dictionary_estimatesL2 = real_tomography(vector, delta=delta, stop_when_reached_accuracy=False,
                                                 norm='L2')
        dictionary_estimatesLinf = real_tomography(vector, delta=delta, stop_when_reached_accuracy=False,
                                                   norm='inf')
        measureL2 = list(dictionary_estimatesL2.keys())
        measureLinf = list(dictionary_estimatesLinf.keys())
        samplesL2 = list(dictionary_estimatesL2.values())
        samplesLinf = list(dictionary_estimatesLinf.values())
        samples_L2_ = [np.linalg.norm(vector - samplesL2[i], ord=2) for i in range(len(samplesL2))]
        samples_Linf_ = [np.linalg.norm(vector - samplesLinf[i], ord=np.inf) for i in range(len(samplesLinf))]

        idx_L2 = np.argwhere(
            np.diff(np.sign(np.full(shape=len(samples_L2_), fill_value=delta) - samples_L2_))).flatten()

        idx_Linf = np.argwhere(
            np.diff(np.sign(np.full(shape=len(samples_Linf_), fill_value=delta) - samples_Linf_))).flatten()

        N_L2 = int((36 * len(vector) * np.log(len(vector))) / (delta ** 2))
        N_Linf = int((36 * np.log(len(vector))) / (delta ** 2))

        fig = plt.figure()

        plt.plot(measureL2[idx_L2[0]], delta, 'ro')
        plt.plot(measureLinf[idx_Linf[0]], delta, 'ro')
        plt.plot(N_L2, delta, 'ro')
        plt.plot(N_Linf, delta, 'ro')
        plt.title("Comparison between real and predicted measurements",
                  fontdict={'family': 'serif',
                            'color': 'darkblue',
                            'weight': 'bold',
                            'size': 18})

        predicted_points_L2 = [compute_predicted_measure(len(vector), i, norm=norm)[0] for i in samples_L2_]
        predicted_points_Linf = [compute_predicted_measure(len(vector), i, norm=norm)[1] for i in samples_Linf_]

        plt.axhline(y=delta, color='k', linestyle='--', label='delta=' + str(delta))
        plt.axvline(x=N_L2, color='purple', linestyle='dotted', label='N_L2=' + str(N_L2))
        plt.axvline(x=N_Linf, color='violet', linestyle='dotted', label='N_Linf=' + str(N_Linf))
        plt.xscale('log')
        plt.plot(predicted_points_L2, samples_L2_, label="theory_L2" + str((N_L2, delta)))
        plt.plot(predicted_points_Linf, samples_Linf_, label="theory_Linf" + str((N_Linf, delta)))
        plt.plot(measureL2, samples_L2_, label="real_L2" + str((measureL2[idx_L2[0]], delta)))
        plt.plot(measureLinf, samples_Linf_, label="real_Linf" + str((measureLinf[idx_Linf[0]], delta)))
        plt.legend()
        # plt.savefig('real_theory_comparison.pdf')
        plt.show()


def found_distribution(vector, n_measurements, delta, distribution_fitter=False, distributions=None, norm='L2',
                       incremental_measure=False, N=None):
    samples = []
    if np.isclose(np.linalg.norm(vector), 1, rtol=1e-2):
        pass
    else:

        vector = vector / np.linalg.norm(vector, ord=2)

    for i in range(n_measurements):
        B = real_tomography(vector, delta=delta, stop_when_reached_accuracy=False, norm=norm,
                            incremental_measure=incremental_measure, N=N)
        print(i)
        # Append the Frobenius norm of A-B to the samples
        B = np.array(list(B.values())[-1])
        if norm == 'L2':
            samples.append(np.linalg.norm(vector - B, ord=2))
        else:
            samples.append(np.linalg.norm(vector - B, ord=np.inf))
    # Plot the samples
    plt.hist(samples, bins=50, color="darkblue")
    plt.xlabel(r"$||\mathbf{v} - \overline{\mathbf{v}}||_2$")
    plt.ylabel("measurements")

    if distribution_fitter:
        f = Fitter(samples, distributions=distributions, timeout=100)
        f.fit()
        f.summary()
        plt.xlabel(r"$||\mathbf{v_{uni}} - \overline{\mathbf{v_{uni}}}||_2$")
        plt.ylabel("measurements")
        print(f.get_best(method='sumsquare_error'))
    # plt.savefig('distribution.pdf')
    plt.show()
    return samples


# Theory vs Real for uniform_vector
#make_real_predicted_comparison(v_uni,0.9)
#decreasing_error_plot(list_, delta=0.9, norm='L2')

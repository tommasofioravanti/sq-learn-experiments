import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import stable_cumsum


def only_dot_product_quantum_new(self, experiment, tests, labels, name_neg_class):
    if experiment == 0:
        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        acc_res = {}
        for e, qpca in enumerate(self.PCAs):
            print(qpca.name)
            rec_list = []
            acc_list = []
            prec_list = []
            f1_score_list = []

            dotted = tests[e].dot(qpca.estimate_right_sv.T)
            sum_major = np.sum(dotted ** 2 / qpca.estimate_fs, axis=1)

            for threshold_major in zip(self.dictionary_major[qpca.name]):
                print(threshold_major)

                attack_predictions = labels[e].iloc[np.where(sum_major > threshold_major)[0]].value_counts()
                normal_predictions = labels[e].iloc[np.where(sum_major <= threshold_major)[0]].value_counts()

                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP

                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]

                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)

                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                rec_list.append(recall)
                prec_list.append(precision)
                acc_list.append(accuracy)

            recall_res.update({qpca.name: rec_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            acc_res.update({qpca.name: acc_list})
        return recall_res, prec_res, acc_res, f1_score_res

    else:

        print('exp1')

        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        accuracy_results = {}

        for e, qpca in enumerate(self.PCAs):

            print(qpca.name)

            print('quantum:', len(qpca.estimate_right_sv), qpca.least_k)
            least_sv = qpca.explained_variance_[qpca.explained_variance_ < self.minorSVvariance]
            nn = np.where(np.isclose(least_sv, 0))[0][0]
            print('classic:', np.searchsorted(stable_cumsum(qpca.explained_variance_ratio_), self.retained_variance[e],
                                              side='right') + 1,len(least_sv[:nn]))
            recall_list = []
            prec_list = []
            f1_score_list = []
            accuracy_list = []

            dotted_major = tests[e].dot(qpca.estimate_right_sv.T)
            dotted_minor = tests[e].dot(qpca.estimate_least_right_sv.T)

            sum_major = np.sum((dotted_major ** 2) / qpca.estimate_fs, axis=1)
            sum_minor = np.sum((dotted_minor ** 2) / qpca.estimate_least_fs, axis=1)

            '''if len((np.where(np.isclose(qpca.estimate_least_fs, 0, atol=1e-4)))[0]) != 0:
                sum_minor = np.sum(
                    dotted_minor.iloc[:,
                    :np.where(np.isclose(qpca.estimate_least_fs, 0, atol=1e-4))[0][0]] ** 2 / qpca.estimate_least_fs[
                                                                                              :np.where(
                                                                                                  np.isclose(
                                                                                                      qpca.estimate_least_fs,
                                                                                                      0, atol=1e-4))[
                                                                                                  0][
                                                                                                  0]],
                    axis=1)
            else:
                sum_minor = np.sum((dotted_minor ** 2) / qpca.estimate_least_fs, axis=1)'''

            for threshold_major, threshold_minor in zip(self.dictionary_major[qpca.name],
                                                        self.dictionary_minor[qpca.name]):

                print(threshold_major, threshold_minor)

                attack_predictions = labels[e].iloc[
                    np.where((sum_major > threshold_major) | (sum_minor > threshold_minor))[0]].value_counts()
                normal_predictions = labels[e].iloc[
                    np.where((sum_major <= threshold_major) & (sum_minor <= threshold_minor))[0]].value_counts()
                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP
                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]
                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)

                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({qpca.name: recall_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            accuracy_results.update({qpca.name: accuracy_list})
        return recall_res, prec_res, accuracy_results, f1_score_res


def only_dot_product_quantum(self, experiment, tests, labels, name_neg_class):
    if experiment == 0:
        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        acc_res = {}
        for e, qpca in enumerate(self.PCAs):
            print(qpca.name)
            rec_list = []
            acc_list = []
            prec_list = []
            f1_score_list = []
            dotted = tests[e].dot(qpca.estimate_right_sv.T)


            '''sum_major = np.sum(
                dotted.iloc[:, :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                side='right') + 1] ** 2 /
                qpca.estimate_fs[
                :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                 side='right') + 1], axis=1)'''
            sum_major = np.sum(dotted ** 2 / qpca.estimate_fs, axis=1)

            for threshold_major in zip(self.dictionary_major[qpca.name]):
                print(threshold_major)

                attack_predictions = labels[e].iloc[np.where(sum_major > threshold_major)[0]].value_counts()
                normal_predictions = labels[e].iloc[np.where(sum_major <= threshold_major)[0]].value_counts()

                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP

                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]

                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)

                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                rec_list.append(recall)
                prec_list.append(precision)
                acc_list.append(accuracy)

            recall_res.update({qpca.name: rec_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            acc_res.update({qpca.name: acc_list})
        return recall_res, prec_res, acc_res, f1_score_res

    else:

        print('exp1')

        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        accuracy_results = {}

        for e, qpca in enumerate(self.PCAs):

            print(qpca.name)

            r = len(qpca.estimate_fs[qpca.estimate_fs < self.minorSVvariance])
            p = len(qpca.estimate_fs)
            print('quantum:', np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                              side='right') + 1, p - r)
            print('classic:', np.searchsorted(stable_cumsum(qpca.explained_variance_ratio_), self.retained_variance[e],
                                              side='right') + 1, len(qpca.explained_variance_) - len(
                qpca.explained_variance_[qpca.explained_variance_ < self.minorSVvariance]))
            recall_list = []
            prec_list = []
            f1_score_list = []
            accuracy_list = []

            TP = 0
            FP = 0
            TN = 0
            FN = 0

            dotted = tests[e].dot(qpca.estimate_right_sv.T)

            sum_major = np.sum(
                dotted.iloc[:, :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                side='right') + 1] ** 2 /
                qpca.estimate_fs[
                :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e], side='right') + 1],
                axis=1)

            if len((np.where(np.isclose(qpca.estimate_fs, 0)))[0]) != 0:
                sum_minor = np.sum(
                    dotted.iloc[:, p - r:np.where(np.isclose(qpca.estimate_fs, 0))[0][0]] ** 2 / qpca.estimate_fs[p - r:
                                                                                                                  np.where(
                                                                                                                      np.isclose(
                                                                                                                          qpca.estimate_fs,
                                                                                                                          0))[
                                                                                                                      0][
                                                                                                                      0]],
                    axis=1)
            else:
                sum_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / qpca.estimate_fs[p - r:], axis=1)

            for threshold_major, threshold_minor in zip(self.dictionary_major[qpca.name],
                                                        self.dictionary_minor[qpca.name]):

                print(threshold_major, threshold_minor)

                attack_predictions = labels[e].iloc[
                    np.where((sum_major > threshold_major) | (sum_minor > threshold_minor))[0]].value_counts()
                normal_predictions = labels[e].iloc[
                    np.where((sum_major <= threshold_major) & (sum_minor <= threshold_minor))[0]].value_counts()
                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP
                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]
                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)

                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({qpca.name: recall_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            accuracy_results.update({qpca.name: accuracy_list})
        return recall_res, prec_res, accuracy_results, f1_score_res


def dot_cosine_corr_measure_quantum_new(self, experiment, tests, labels, name_neg_class):
    if experiment == 0:
        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        acc_res = {}
        for e, qpca in enumerate(self.PCAs):

            print(qpca.name)
            rec_list = []
            acc_list = []
            prec_list = []
            f1_score_list = []
            sum_major_corr=[]

            dotted = tests[e].dot(qpca.estimate_right_sv.T)
            sum_major = np.sum(dotted ** 2 / qpca.estimate_fs, axis=1)

            try:
                dataframe_cosine = cosine_similarity(tests[e], qpca.estimate_right_sv)

            except:
                dataframe_cosine = cosine_similarity(np.nan_to_num(tests[e]), qpca.estimate_right_sv)

            for j in range(len(tests[e])):
                try:
                    y_corr_maj = np.corrcoef(tests[e].iloc[j], qpca.estimate_right_sv)[0][1:]
                except:
                    y_corr_maj = np.corrcoef(np.nan_to_num(tests[e].iloc[j]), qpca.estimate_right_sv)[0][1:]

                sum_major_corr.append(np.sum((y_corr_maj ** 2) / qpca.estimate_fs))
            sum_major_cosine = np.sum(dataframe_cosine ** 2 / qpca.estimate_fs, axis=1)

            for threshold_major, threshold_major_cosine, threshold_major_corr in zip(self.dictionary_major[qpca.name],
                                                                                     self.dictionary_major_cosine[
                                                                                         qpca.name],
                                                                                     self.dictionary_major_corr[
                                                                                         qpca.name]):
                print(threshold_major, threshold_major_cosine, threshold_major_corr)

                attack_predictions = labels[e].iloc[np.where((sum_major > threshold_major) |
                                                             (sum_major_cosine > threshold_major_cosine) |
                                                             (sum_major_corr > threshold_major_corr))[0]].value_counts()
                normal_predictions = labels[e].iloc[
                    np.where((sum_major <= threshold_major) & (sum_major_cosine <= threshold_major_cosine)
                             & (sum_major_corr <= threshold_major_corr))[0]].value_counts()

                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP
                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]
                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)
                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                rec_list.append(recall)
                prec_list.append(precision)
                acc_list.append(accuracy)

            recall_res.update({qpca.name: rec_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            acc_res.update({qpca.name: acc_list})

        return recall_res, prec_res, acc_res, f1_score_res

    else:

        print('exp1')
        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        accuracy_results = {}

        for e, qpca in enumerate(self.PCAs):
            sum_major_corr = []
            sum_minor_corr = []

            print(qpca.name)
            recall_list = []
            prec_list = []
            f1_score_list = []
            accuracy_list = []

            print('quantum:', len(qpca.estimate_right_sv), qpca.least_k)
            least_sv = qpca.explained_variance_[qpca.explained_variance_ < self.minorSVvariance]
            nn = np.where(np.isclose(least_sv, 0))[0][0]
            print('classic:', np.searchsorted(stable_cumsum(qpca.explained_variance_ratio_), self.retained_variance[e],
                                              side='right') + 1, len(least_sv[:nn]))

            dotted_major = tests[e].dot(qpca.estimate_right_sv.T)
            dotted_minor = tests[e].dot(qpca.estimate_least_right_sv.T)
            try:
                dataframe_cosine_major = cosine_similarity(tests[e], qpca.estimate_right_sv)
                dataframe_cosine_minor = cosine_similarity(tests[e], qpca.estimate_least_right_sv)
            except:
                dataframe_cosine_major = cosine_similarity(np.nan_to_num(tests[e]), qpca.estimate_right_sv)
                dataframe_cosine_minor = cosine_similarity(np.nan_to_num(tests[e]), qpca.estimate_least_right_sv)

            sum_major = np.sum((dotted_major ** 2) / qpca.estimate_fs, axis=1)

            sum_major_cosine = np.sum((dataframe_cosine_major ** 2) / qpca.estimate_fs, axis=1)

            for j in range(len(tests[e])):
                try:
                    y_corr_maj = np.corrcoef(tests[e].iloc[j], qpca.estimate_right_sv)[0][1:]
                    y_corr_min = np.corrcoef(tests[e].iloc[j], qpca.estimate_least_right_sv)[0][1:]
                except:
                    y_corr_maj = np.corrcoef(np.nan_to_num(tests[e].iloc[j]), qpca.estimate_right_sv)[0][1:]
                    y_corr_min = np.corrcoef(np.nan_to_num(tests[e].iloc[j]), qpca.estimate_least_right_sv)[0][1:]

                sum_major_corr.append(np.sum((y_corr_maj ** 2) / qpca.estimate_fs))
                sum_minor_corr.append(np.sum((y_corr_min ** 2) / qpca.estimate_least_fs))

                '''if len(np.where(np.isclose(qpca.estimate_least_fs, 0, atol=1e-4))[0]) != 0:
                    sum_minor_corr.append(
                        np.sum((y_corr_min[:np.where(np.isclose(qpca.estimate_least_fs, 0, atol=1e-4))[0][0]] ** 2) /
                               qpca.estimate_least_fs[
                               :np.where(np.isclose(qpca.estimate_least_fs, 0, atol=1e-4))[0][0]]))
                else:
                    sum_minor_corr.append(np.sum((y_corr_min ** 2) / qpca.estimate_least_fs))'''

            sum_major_corr = np.array(sum_major_corr)
            sum_minor_corr = np.array(sum_minor_corr)

            sum_minor = np.sum((dotted_minor ** 2) / qpca.estimate_least_fs, axis=1)
            sum_minor_cosine = np.sum((dataframe_cosine_minor ** 2) / qpca.estimate_least_fs, axis=1)

            '''if len((np.where(np.isclose(qpca.estimate_least_fs, 0,atol=1e-4)))[0]) != 0:
                sum_minor = np.sum((
                    dotted_minor.iloc[:,:np.where(np.isclose(qpca.estimate_least_fs, 0,atol=1e-4))[0][0]] ** 2) /
                    qpca.estimate_least_fs[:np.where(np.isclose(qpca.estimate_least_fs, 0,atol=1e-4))[0][0]], axis=1)
                sum_minor_cosine = np.sum((
                    dataframe_cosine_minor[:,:np.where(np.isclose(qpca.estimate_least_fs, 0,atol=1e-4))[0][0]] ** 2 )/
                    qpca.estimate_least_fs[:np.where(np.isclose(qpca.estimate_least_fs, 0,atol=1e-4))[0][0]], axis=1)
            else:
                sum_minor = np.sum((dotted_minor ** 2) / qpca.estimate_least_fs, axis=1)
                sum_minor_cosine = np.sum((dataframe_cosine_minor ** 2) / qpca.estimate_least_fs, axis=1)'''

            for threshold_major, threshold_minor, threshold_major_cosine, threshold_minor_cosine, threshold_major_corr, threshold_minor_corr in zip(
                    self.dictionary_major[qpca.name], self.dictionary_minor[qpca.name],
                    self.dictionary_major_cosine[qpca.name],
                    self.dictionary_minor_cosine[qpca.name], self.dictionary_major_corr[qpca.name],
                    self.dictionary_minor_corr[qpca.name]):

                print(threshold_major, threshold_minor, threshold_major_cosine, threshold_minor_cosine,
                      threshold_major_corr, threshold_minor_corr)
                kk=np.where((sum_major > threshold_major) | (sum_major_cosine > threshold_major_cosine)
                             | (sum_minor > threshold_minor) | (sum_minor_cosine > threshold_minor_cosine)
                             | (sum_major_corr > threshold_major_corr) | (sum_minor_corr > threshold_minor_corr))[
                        0]
                attack_predictions = labels[e].iloc[
                    np.where((sum_major > threshold_major) | (sum_major_cosine > threshold_major_cosine)
                             | (sum_minor > threshold_minor) | (sum_minor_cosine > threshold_minor_cosine)
                             | (sum_major_corr > threshold_major_corr) | (sum_minor_corr > threshold_minor_corr))[
                        0]].value_counts()
                normal_predictions = labels[e].iloc[
                    np.where((sum_major <= threshold_major) & (sum_major_cosine <= threshold_major_cosine)
                             & (sum_minor <= threshold_minor) & (sum_minor_cosine <= threshold_minor_cosine)
                             & (sum_major_corr <= threshold_major_corr) & (sum_minor_corr <= threshold_minor_corr))[
                        0]].value_counts()

                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP
                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]
                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)
                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({qpca.name: recall_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            accuracy_results.update({qpca.name: accuracy_list})

        return recall_res, prec_res, accuracy_results, f1_score_res


def dot_cosine_corr_measure_quantum(self, experiment, tests, labels, name_neg_class):
    if experiment == 0:
        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        acc_res = {}
        for e, qpca in enumerate(self.PCAs):

            print(qpca.name)
            rec_list = []
            acc_list = []
            prec_list = []
            f1_score_list = []

            TP = 0
            FP = 0
            TN = 0
            FN = 0

            dotted = tests[e].dot(qpca.estimate_right_sv.T)
            try:
                dataframe_cosine = cosine_similarity(tests[e], qpca.estimate_right_sv)
                dataframe_corr = np.corrcoef(tests[e], qpca.estimate_right_sv)
            except:
                dataframe_cosine = cosine_similarity(np.nan_to_num(tests[e]), qpca.estimate_right_sv)
                dataframe_corr = np.corrcoef(np.nan_to_num(tests[e]), qpca.estimate_right_sv)

            sum_major = np.sum(dotted.iloc[:,
                               :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                side='right') + 1] ** 2 / qpca.estimate_fs[
                                                                          :np.searchsorted(
                                                                              stable_cumsum(qpca.estimate_fs_ratio),
                                                                              self.retained_variance[e],
                                                                              side='right') + 1], axis=1)

            sum_major_corr = np.sum(dataframe_corr[:len(tests[e]),
                                    len(tests[e]):len(tests[e]) + np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio),
                                                                                  self.retained_variance[e],
                                                                                  side='right') + 1] ** 2 /
                                    qpca.estimate_fs[
                                    :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                     side='right') + 1], axis=1)

            sum_major_cosine = np.sum(dataframe_cosine[:,
                                      :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                       side='right') + 1] ** 2 /
                                      qpca.estimate_fs[
                                      :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                       side='right') + 1], axis=1)

            for threshold_major, threshold_major_cosine, threshold_major_corr in zip(self.dictionary_major[qpca.name],
                                                                                     self.dictionary_major_cosine[
                                                                                         qpca.name],
                                                                                     self.dictionary_major_corr[
                                                                                         qpca.name]):
                print(threshold_major, threshold_major_cosine, threshold_major_corr)

                attack_predictions = labels[e].iloc[np.where((sum_major > threshold_major) |
                                                             (sum_major_cosine > threshold_major_cosine) |
                                                             (sum_major_corr > threshold_major_corr))[0]].value_counts()
                normal_predictions = labels[e].iloc[
                    np.where((sum_major <= threshold_major) & (sum_major_cosine <= threshold_major_cosine)
                             & (sum_major_corr <= threshold_major_corr))[0]].value_counts()

                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP
                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]
                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)
                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                rec_list.append(recall)
                prec_list.append(precision)
                acc_list.append(accuracy)

            recall_res.update({qpca.name: rec_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            acc_res.update({qpca.name: acc_list})

        return recall_res, prec_res, acc_res, f1_score_res

    else:

        print('exp1')

        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        accuracy_results = {}

        for e, qpca in enumerate(self.PCAs):
            sum_major_corr = []
            sum_minor_corr = []

            print(qpca.name)

            r = len(qpca.estimate_fs[qpca.estimate_fs < self.minorSVvariance])
            p = len(qpca.estimate_fs)

            recall_list = []
            prec_list = []
            f1_score_list = []
            accuracy_list = []

            TP = 0
            FP = 0
            TN = 0
            FN = 0
            print('quantum:', np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                              side='right') + 1, p - r)
            print('classic:', np.searchsorted(stable_cumsum(qpca.explained_variance_ratio_), self.retained_variance[e],
                                              side='right') + 1, len(qpca.explained_variance_) - len(
                qpca.explained_variance_[qpca.explained_variance_ < self.minorSVvariance]))

            dotted = tests[e].dot(qpca.estimate_right_sv.T)
            try:
                dataframe_cosine = cosine_similarity(tests[e], qpca.estimate_right_sv)
            except:
                dataframe_cosine = cosine_similarity(np.nan_to_num(tests[e]), qpca.estimate_right_sv)

            sum_major = np.sum(
                dotted.iloc[:, :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                side='right') + 1] ** 2 /
                qpca.estimate_fs[
                :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e], side='right') + 1],
                axis=1)

            sum_major_cosine = np.sum(dataframe_cosine[:,
                                      :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                       side='right') + 1] ** 2 /
                                      qpca.estimate_fs[
                                      :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio), self.retained_variance[e],
                                                       side='right') + 1], axis=1)

            for j in range(len(tests[e])):
                try:
                    y_corr = np.corrcoef(tests[e].iloc[j], qpca.estimate_right_sv)[0][1:]
                except:
                    y_corr = np.corrcoef(np.nan_to_num(tests[e].iloc[j]), qpca.estimate_right_sv)[0][1:]

                sum_major_corr.append(np.sum(y_corr[:np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio),
                                                                     self.retained_variance[e],
                                                                     side='right') + 1] ** 2 /
                                             qpca.estimate_fs[
                                             :np.searchsorted(stable_cumsum(qpca.estimate_fs_ratio),
                                                              self.retained_variance[e], side='right') + 1]))
                if len(np.where(np.isclose(qpca.estimate_fs, 0))[0]) != 0:
                    sum_minor_corr.append(np.sum(y_corr[p - r:np.where(np.isclose(qpca.estimate_fs, 0))[0][0]] ** 2 /
                                                 qpca.estimate_fs[
                                                 p - r:np.where(np.isclose(qpca.estimate_fs, 0))[0][0]]))
                else:
                    sum_minor_corr.append(np.sum(y_corr[p - r:] ** 2 / qpca.estimate_fs[p - r:]))

            sum_major_corr = np.array(sum_major_corr)
            sum_minor_corr = np.array(sum_minor_corr)

            if len((np.where(np.isclose(qpca.estimate_fs, 0)))[0]) != 0:
                sum_minor = np.sum(
                    dotted.iloc[:, p - r:np.where(np.isclose(qpca.estimate_fs, 0))[0][0]] ** 2 /
                    qpca.estimate_fs[p - r:np.where(np.isclose(qpca.estimate_fs, 0))[0][0]], axis=1)
                sum_minor_cosine = np.sum(
                    dataframe_cosine[:, p - r:np.where(np.isclose(qpca.estimate_fs, 0))[0][0]] ** 2 /
                    qpca.estimate_fs[p - r:np.where(np.isclose(qpca.estimate_fs, 0))[0][0]], axis=1)
            else:
                sum_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / qpca.estimate_fs[p - r:], axis=1)
                sum_minor_cosine = np.sum(dataframe_cosine[:, p - r:] ** 2 / qpca.estimate_fs[p - r:], axis=1)

            for threshold_major, threshold_minor, threshold_major_cosine, threshold_minor_cosine, threshold_major_corr, threshold_minor_corr in zip(
                    self.dictionary_major[qpca.name], self.dictionary_minor[qpca.name],
                    self.dictionary_major_cosine[qpca.name],
                    self.dictionary_minor_cosine[qpca.name], self.dictionary_major_corr[qpca.name],
                    self.dictionary_minor_corr[qpca.name]):

                print(threshold_major, threshold_minor, threshold_major_cosine, threshold_minor_cosine,
                      threshold_major_corr, threshold_minor_corr)

                attack_predictions = labels[e].iloc[
                    np.where((sum_major > threshold_major) | (sum_major_cosine > threshold_major_cosine)
                             | (sum_minor > threshold_minor) | (sum_minor_cosine > threshold_minor_cosine)
                             | (sum_major_corr > threshold_major_corr) | (sum_minor_corr > threshold_minor_corr))[
                        0]].value_counts()
                normal_predictions = labels[e].iloc[
                    np.where((sum_major <= threshold_major) & (sum_major_cosine <= threshold_major_cosine)
                             & (sum_minor <= threshold_minor) & (sum_minor_cosine <= threshold_minor_cosine)
                             & (sum_major_corr <= threshold_major_corr) & (sum_minor_corr <= threshold_minor_corr))[
                        0]].value_counts()

                total_predicted_attack = attack_predictions.sum()

                FP = attack_predictions[name_neg_class]
                TP = total_predicted_attack - FP
                total_predicted_negative = normal_predictions.sum()

                TN = normal_predictions[name_neg_class]
                FN = total_predicted_negative - TN

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                print('detection_rate:', recall)
                print('precision:', precision)
                print('accuracy:', accuracy)
                if precision > 0 and recall > 0:
                    f1_score = 2 / ((1 / recall) + (1 / precision))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({qpca.name: recall_list})
            prec_res.update({qpca.name: prec_list})
            f1_score_res.update({qpca.name: f1_score_list})
            accuracy_results.update({qpca.name: accuracy_list})

        return recall_res, prec_res, accuracy_results, f1_score_res

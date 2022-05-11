import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def only_dot_prod_normal(self, experiment, tests, labels, name_neg_class):
    if experiment == 0:

        recall_res = {}
        prec_res = {}
        acc_res = {}
        f1_score_res = {}

        for e, pca in enumerate(self.PCAs):

            print(pca.name)
            recall_list = []
            prec_list = []
            accuracy_list = []
            f1_score_list = []

            TP = 0
            FP = 0
            TN = 0
            FN = 0

            dotted = tests[e].dot(pca.components_.T)
            sum_major = np.sum(dotted.iloc[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                axis=1)
            for threshold_major in zip(self.dictionary_major[pca.name]):
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
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({pca.name: recall_list})
            prec_res.update({pca.name: prec_list})
            f1_score_res.update({pca.name: f1_score_list})
            acc_res.update({pca.name: recall_list})

        return recall_res, prec_res, acc_res, f1_score_res

    else:

        print('exp1')
        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        accuracy_results = {}

        for e, pca in enumerate(self.PCAs):

            print(pca.name)
            r = len(pca.explained_variance_[pca.explained_variance_ < self.minorSVvariance])
            p = len(pca.explained_variance_)

            recall_list = []
            prec_list = []
            f1_score_list = []
            accuracy_list = []

            TP = 0
            FP = 0
            TN = 0
            FN = 0

            dotted = tests[e].dot(pca.components_.T)
            sum_major = np.sum(dotted.iloc[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                axis=1)

            if len(np.where(np.isclose(pca.explained_variance_, 0))[0]) != 0 :
                sum_minor = np.sum(dotted.iloc[:, p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]] ** 2 / pca.explained_variance_[
                               p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]], axis=1)

            else:
                sum_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / pca.explained_variance_[p - r:], axis=1)

            for threshold_major, threshold_minor in zip(self.dictionary_major[pca.name],
                                                        self.dictionary_minor[pca.name]):

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
                    f1_score = 2 / ((1 / precision) + (1 / recall))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP + FP + TN + FN, 'LEN_TEST:', len(tests[e]))
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({pca.name: recall_list})
            prec_res.update({pca.name: prec_list})
            f1_score_res.update({pca.name: f1_score_list})
            accuracy_results.update({pca.name: accuracy_list})

        return recall_res, prec_res, accuracy_results, f1_score_res


def dot_cosine_corr_measure_normal(self, experiment, labels, tests, name_neg_class):
    if experiment == 0:
        recall_res = {}
        prec_res = {}
        acc_res = {}
        f1_score_res = {}
        for e, pca in enumerate(self.PCAs):
            print(pca.name)
            sum_major_corr = []
            recall_list = []
            prec_list = []
            accuracy_list = []
            f1_score_list = []

            TP = 0
            FP = 0
            TN = 0
            FN = 0

            dotted = tests[e].dot(pca.components_.T)
            try:
                cosine_together = cosine_similarity(tests[e], pca.components_)
            except:
                cosine_together = cosine_similarity(np.nan_to_num(tests[e]), pca.components_)

            sum_major = np.sum(dotted.iloc[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                axis=1)

            sum_major_cosine = np.sum(cosine_together[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                                      axis=1)
            for j in range(len(tests[e])):
                try:
                    y_corr = np.corrcoef(tests[e].iloc[j], pca.components_)[0][1:]
                except:
                    y_corr = np.corrcoef(np.nan_to_num(tests[e].iloc[j]), pca.components_)[0][1:]

                sum_major_corr.append(np.sum(y_corr[:pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_]))

            sum_major_corr = np.array(sum_major_corr)

            for threshold_major, threshold_major_cosine, threshold_major_corr in zip(
                    self.dictionary_major[pca.name], self.dictionary_major_cosine[pca.name],
                    self.dictionary_major_corr[pca.name]):
                print(threshold_major, threshold_major_cosine, threshold_major_corr)

                attack_predictions = labels[e].iloc[np.where((sum_major > threshold_major) |
                                                             (sum_major_cosine > threshold_major_cosine) |
                                                             (sum_major_corr > threshold_major_corr))[0]].value_counts()
                normal_predictions = labels[e].iloc[
                    np.where((sum_major <= threshold_major) & (sum_major_cosine <= threshold_major_cosine)
                             &(sum_major_corr <= threshold_major_corr))[0]].value_counts()

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
                      TP + FP+TN +FN, 'LEN_TEST:', len(tests[e]))
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({pca.name: recall_list})
            prec_res.update({pca.name: prec_list})
            f1_score_res.update({pca.name: f1_score_list})
            acc_res.update({pca.name: recall_list})
        return recall_res, prec_res, acc_res, f1_score_res

    else:

        print('exp1')

        recall_res = {}
        prec_res = {}
        f1_score_res = {}
        accuracy_results = {}

        for e, pca in enumerate(self.PCAs):
            sum_major_corr = []
            sum_minor_corr = []
            print(pca.name)
            r = len(pca.explained_variance_[pca.explained_variance_ < self.minorSVvariance])
            p = len(pca.explained_variance_)

            recall_list = []
            prec_list = []
            f1_score_list = []
            accuracy_list = []

            TP = 0
            FP = 0
            TN = 0
            FN = 0

            dotted = tests[e].dot(pca.components_.T)
            try:

                cosine_together = cosine_similarity(tests[e], pca.components_)
            except:
                cosine_together = cosine_similarity(np.nan_to_num(tests[e]), pca.components_)

            sum_major = np.sum(dotted.iloc[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                axis=1)

            sum_major_cosine = np.sum(cosine_together[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                                      axis=1)
            for j in range(len(tests[e])):
                try:
                    y_corr = np.corrcoef(tests[e].iloc[j], pca.components_)[0][1:]
                except:
                    y_corr = np.corrcoef(np.nan_to_num(tests[e].iloc[j]), pca.components_)[0][1:]

                sum_major_corr.append(np.sum(y_corr[:pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_]))
                if len(np.where(np.isclose(pca.explained_variance_, 0))[0]) != 0:
                    sum_minor_corr.append(np.sum(y_corr[p-r:np.where(np.isclose(pca.explained_variance_,0))[0][0]]**2 / pca.explained_variance_[p-r:np.where(np.isclose(pca.explained_variance_,0))[0][0]]))
                else:
                    sum_minor_corr.append(np.sum(y_corr[p-r:]**2 / pca.explained_variance_[p-r:]))

            sum_major_corr = np.array(sum_major_corr)
            sum_minor_corr = np.array(sum_minor_corr)

            if len(np.where(np.isclose(pca.explained_variance_, 0))[0]) != 0:

                sum_minor = np.sum(dotted.iloc[:, p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]] ** 2 / pca.explained_variance_[p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]], axis=1)

                sum_minor_cosine = np.sum(cosine_together[:, p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]] ** 2 / pca.explained_variance_[
                                                         p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]],
                                          axis=1)
            else:

                sum_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / pca.explained_variance_[p - r:], axis=1)

                sum_minor_cosine = np.sum(cosine_together[:, p - r:] ** 2 / pca.explained_variance_[p - r:], axis=1)

            for threshold_major, threshold_minor, threshold_major_cosine, threshold_minor_cosine, threshold_major_corr, threshold_minor_corr in zip(
                    self.dictionary_major[pca.name], self.dictionary_minor[pca.name],
                    self.dictionary_major_cosine[pca.name],
                    self.dictionary_minor_cosine[pca.name], self.dictionary_major_corr[pca.name],
                    self.dictionary_minor_corr[pca.name]):

                print(threshold_major, threshold_minor, threshold_major_cosine, threshold_minor_cosine,threshold_major_corr,threshold_minor_corr)

                attack_predictions = labels[e].iloc[np.where((sum_major > threshold_major) | (sum_major_cosine > threshold_major_cosine)
                                                             | (sum_minor > threshold_minor) | (sum_minor_cosine > threshold_minor_cosine)
                                                             | (sum_major_corr > threshold_major_corr) | (sum_minor_corr > threshold_minor_corr))[0]].value_counts()
                normal_predictions = labels[e].iloc[np.where((sum_major <= threshold_major) & (sum_major_cosine <= threshold_major_cosine)
                                                             & (sum_minor <= threshold_minor) & (sum_minor_cosine <= threshold_minor_cosine)
                                                             &(sum_major_corr <= threshold_major_corr) & (sum_minor_corr <= threshold_minor_corr))[0]].value_counts()

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
                    f1_score = 2 / ((1 / precision) + (1 / recall))
                    print('F1_score:', f1_score)
                    f1_score_list.append(f1_score)

                print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, 'TOT_SAMPLES:',
                      TP+ FP +TN +FN, 'LEN_TEST:', len(tests[e]))
                recall_list.append(recall)
                prec_list.append(precision)
                accuracy_list.append(accuracy)

            recall_res.update({pca.name: recall_list})
            prec_res.update({pca.name: prec_list})
            f1_score_res.update({pca.name: f1_score_list})
            accuracy_results.update({pca.name: accuracy_list})

        return recall_res, prec_res, accuracy_results, f1_score_res

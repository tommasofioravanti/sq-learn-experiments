import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import stable_cumsum
import pandas as pd


def fit_new_quantum(self, training_sets, minor_sv_variance, only_dot_product, experiment):
    self.minorSVvariance = minor_sv_variance
    dictionary_major = {}
    dictionary_minor = {}
    ret_variance = [0.30, 0.40, 0.50, 0.60, 0.70]
    self.retained_variance = ret_variance

    if only_dot_product:
        for e, pca in enumerate(self.PCAs):
            out_threshold_list_major = []
            out_threshold_list_minor = []

            dotted_major = training_sets[e].dot(pca.estimate_right_sv.T)
            if experiment == 1:
                dotted_minor = training_sets[e].dot(pca.estimate_least_right_sv.T)

            if isinstance(dotted_major, pd.DataFrame):
                s_major = np.sum((dotted_major ** 2) / (pca.estimate_fs), axis=1)

            if experiment == 1:
                s_minor = np.sum((dotted_minor ** 2) / (pca.estimate_least_fs), axis=1)

                '''if len((np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4)))[0]) != 0:
                    if isinstance(dotted_major, pd.DataFrame):
                        s_minor = np.sum(dotted_minor.iloc[:, : np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][
                                0]] ** 2 / pca.estimate_least_fs[:np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][0]], axis=1)
                else:
                    s_minor = np.sum((dotted_minor ** 2) / (pca.estimate_least_fs), axis=1)'''

            if isinstance(dotted_major, pd.DataFrame):
                emp_distribution_major = s_major.values
                if experiment == 1:
                    emp_distribution_minor = s_minor.values
            else:
                emp_distribution_major = s_major
                if experiment == 1:
                    emp_distribution_minor = s_minor

            for _, q in enumerate(self.quantils):
                n_major = len(emp_distribution_major)
                sort_major = sorted(emp_distribution_major)
                out_threshold_major = sort_major[int(n_major * q)]
                out_threshold_list_major.append(out_threshold_major)
                if experiment == 1:
                    n_minor = len(emp_distribution_minor)
                    sort_minor = sorted(emp_distribution_minor)
                    out_threshold_minor = sort_minor[int(n_minor * q)]
                    out_threshold_list_minor.append(out_threshold_minor)

            dictionary_major.update({pca.name: out_threshold_list_major})
            if experiment == 1:
                dictionary_minor.update({pca.name: out_threshold_list_minor})
        self.dictionary_major = dictionary_major
        if experiment == 1:
            self.dictionary_minor = dictionary_minor

    else:
        dictionary_major_corr = {}
        dictionary_minor_corr = {}

        dictionary_major_cosine = {}
        dictionary_minor_cosine = {}

        for e, pca in enumerate(self.PCAs):

            out_threshold_list_major = []
            out_threshold_list_minor = []

            out_threshold_list_major_cosine = []
            out_threshold_list_minor_cosine = []

            out_threshold_list_major_corr = []
            out_threshold_list_minor_corr = []

            emp_distribution_major_corr = []
            emp_distribution_minor_corr = []

            dotted_major = training_sets[e].dot(pca.estimate_right_sv.T)
            dotted_minor = training_sets[e].dot(pca.estimate_least_right_sv.T)
            if isinstance(dotted_major, pd.DataFrame):
                s_major = np.sum(dotted_major ** 2 / pca.estimate_fs, axis=1)

            cosine_together_major = cosine_similarity(training_sets[e], pca.estimate_right_sv)
            cosine_together_minor = cosine_similarity(training_sets[e], pca.estimate_least_right_sv)

            s_major_cosine = np.sum((cosine_together_major ** 2) / pca.estimate_fs, axis=1)

            s_minor = np.sum((dotted_minor ** 2) / pca.estimate_least_fs, axis=1)

            s_minor_cosine = np.sum((cosine_together_minor ** 2) / pca.estimate_least_fs, axis=1)

            '''if len((np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4)))[0]) != 0:
                if isinstance(dotted_minor, pd.DataFrame):
                    s_minor = np.sum(
                        dotted_minor.iloc[:, : np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][
                            0]] ** 2 / pca.estimate_least_fs[
                                       :np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][0]], axis=1)
                else:
                    s_minor = np.sum(dotted_minor[:, : np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][
                        0]] ** 2 / pca.estimate_least_fs[
                                   :np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][0]], axis=1)
                s_minor_cosine = np.sum(
                    cosine_together_minor[:, :np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][
                        0]] ** 2 / pca.estimate_least_fs[:np.where(np.isclose(pca.estimate_fs, 0, atol=1e-4))[0][0]],
                    axis=1)
            else:
                s_minor = np.sum((dotted_minor ** 2) / pca.estimate_least_fs, axis=1)

                s_minor_cosine = np.sum((cosine_together_minor ** 2) / pca.estimate_least_fs, axis=1)'''
            for j in range(len(training_sets[e])):
                if isinstance(training_sets[e], pd.DataFrame):
                    y_corr_maj = np.corrcoef(training_sets[e].iloc[j], pca.estimate_right_sv)[0][1:]
                    y_corr_min = np.corrcoef(training_sets[e].iloc[j], pca.estimate_least_right_sv)[0][1:]
                else:
                    y_corr_maj = np.corrcoef(training_sets[e][j], pca.estimate_right_sv)[0][1:]
                    y_corr_min = np.corrcoef(training_sets[e][j], pca.estimate_least_right_sv)[0][1:]

                s_major_corr = np.sum((y_corr_maj ** 2) / pca.estimate_fs)
                s_minor_corr = np.sum((y_corr_min ** 2) / pca.estimate_least_fs)

                '''if len((np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4)))[0]) != 0:
                    s_minor_corr = np.sum((y_corr_min[:np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][
                        0]] ** 2) / pca.estimate_least_fs[
                                    :np.where(np.isclose(pca.estimate_least_fs, 0, atol=1e-4))[0][0]])
                else:
                    s_minor_corr = np.sum((y_corr_min ** 2) / pca.estimate_least_fs)'''

                emp_distribution_major_corr.append(s_major_corr)
                emp_distribution_minor_corr.append(s_minor_corr)

            if isinstance(dotted_major, pd.DataFrame):
                emp_distribution_major = s_major.values
                emp_distribution_minor = s_minor.values
            else:
                emp_distribution_major = s_major
                emp_distribution_minor = s_minor
            emp_distribution_major_cosine = s_major_cosine
            emp_distribution_minor_cosine = s_minor_cosine

            for _, q in enumerate(self.quantils):
                n_major = len(emp_distribution_major)
                n_minor = len(emp_distribution_minor)

                n_major_corr = len(emp_distribution_major_corr)
                n_minor_corr = len(emp_distribution_minor_corr)

                n_major_cosine = len(emp_distribution_major_cosine)
                n_minor_cosine = len(emp_distribution_minor_cosine)

                sort_major = sorted(emp_distribution_major)
                sort_minor = sorted(emp_distribution_minor)

                sort_major_corr = sorted(emp_distribution_major_corr)
                sort_minor_corr = sorted(emp_distribution_minor_corr)

                sort_major_cosine = sorted(emp_distribution_major_cosine)
                sort_minor_cosine = sorted(emp_distribution_minor_cosine)

                out_threshold_major = sort_major[int(n_major * q)]
                out_threshold_minor = sort_minor[int(n_minor * q)]

                out_threshold_major_corr = sort_major_corr[int(n_major_corr * q)]
                out_threshold_minor_corr = sort_minor_corr[int(n_minor_corr * q)]

                out_threshold_major_cosine = sort_major_cosine[int(n_major_cosine * q)]
                out_threshold_minor_cosine = sort_minor_cosine[int(n_minor_cosine * q)]

                out_threshold_list_major.append(out_threshold_major)
                out_threshold_list_minor.append(out_threshold_minor)

                out_threshold_list_major_corr.append(out_threshold_major_corr)
                out_threshold_list_minor_corr.append(out_threshold_minor_corr)

                out_threshold_list_major_cosine.append(out_threshold_major_cosine)
                out_threshold_list_minor_cosine.append(out_threshold_minor_cosine)

            dictionary_major.update({pca.name: out_threshold_list_major})
            dictionary_minor.update({pca.name: out_threshold_list_minor})

            dictionary_major_corr.update({pca.name: out_threshold_list_major_corr})
            dictionary_minor_corr.update({pca.name: out_threshold_list_minor_corr})

            dictionary_major_cosine.update({pca.name: out_threshold_list_major_cosine})
            dictionary_minor_cosine.update({pca.name: out_threshold_list_minor_cosine})

        self.dictionary_major = dictionary_major
        self.dictionary_minor = dictionary_minor

        self.dictionary_major_corr = dictionary_major_corr
        self.dictionary_minor_corr = dictionary_minor_corr

        self.dictionary_major_cosine = dictionary_major_cosine
        self.dictionary_minor_cosine = dictionary_minor_cosine

    return self


def fit_quantum(self, training_sets, minor_sv_variance, only_dot_product, experiment):
    self.minorSVvariance = minor_sv_variance
    dictionary_major = {}
    dictionary_minor = {}
    dictionary_major_classic = {}
    ret_variance = [0.30, 0.40, 0.50, 0.60, 0.70]
    self.retained_variance = ret_variance

    if only_dot_product:
        for e, pca in enumerate(self.PCAs):

            r = len(pca.estimate_fs[pca.estimate_fs < self.minorSVvariance])
            p = len(pca.estimate_fs)

            out_threshold_list_major = []
            out_threshold_list_minor = []
            out_threshold_list_major_classic = []

            dotted = training_sets[e].dot(pca.estimate_right_sv.T)
            dotted_classic = training_sets[e].dot(pca.topk_right_singular_vectors.T)
            # print(np.dot(training_sets[e].iloc[0], pca.estimate_right_sv.T))
            if isinstance(dotted, pd.DataFrame):
                if experiment == 0:
                    s_major = np.sum((dotted ** 2) / (
                        pca.estimate_fs),
                                     axis=1)
                    s_major_classic = np.sum((dotted_classic ** 2) / pca.explained_variance_[:pca.topk], axis=1)
                else:
                    print(np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                          self.retained_variance[e],
                                          side='right') + 1, p - r)
                    s_major = np.sum((dotted.iloc[:, :np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                      self.retained_variance[e],
                                                                      side='right') + 1] ** 2) / (
                                         pca.estimate_fs[:np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                          self.retained_variance[e],
                                                                          side='right') + 1]),
                                     axis=1)


            else:
                if experiment == 0:
                    s_major = np.sum((dotted ** 2) / (
                        pca.estimate_fs),
                                     axis=1)
                else:
                    s_major = np.sum((dotted[:, :np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                 self.retained_variance[e], side='right') + 1] ** 2) /
                                     (pca.estimate_fs[:np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                       self.retained_variance[e], side='right') + 1]),
                                     axis=1)

            if len((np.where(np.isclose(pca.estimate_fs, 0)))[0]) != 0:
                if isinstance(dotted, pd.DataFrame):

                    s_minor = np.sum(dotted.iloc[:, p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]] ** 2 /
                                     pca.estimate_fs[p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]], axis=1)

                else:

                    s_minor = np.sum(dotted[:, p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]] ** 2 /
                                     pca.estimate_fs[p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]], axis=1)
            else:

                if isinstance(dotted, pd.DataFrame):
                    s_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / pca.estimate_fs[p - r:], axis=1)
                else:
                    s_minor = np.sum(dotted[:, p - r:] ** 2 / pca.estimate_fs[p - r:], axis=1)

            if isinstance(dotted, pd.DataFrame):
                emp_distribution_major = s_major.values
                # emp_distribution_major_classic = s_major_classic.values
                emp_distribution_minor = s_minor.values


            else:
                emp_distribution_major = s_major
                emp_distribution_minor = s_minor

            for _, q in enumerate(self.quantils):
                n_major = len(emp_distribution_major)
                # n_major_classic = len(emp_distribution_major_classic)
                n_minor = len(emp_distribution_minor)

                sort_major = sorted(emp_distribution_major)
                # sort_major_classic = sorted(emp_distribution_major_classic)
                sort_minor = sorted(emp_distribution_minor)

                out_threshold_major = sort_major[int(n_major * q)]
                # out_threshold_major_classic = sort_major_classic[int(n_major_classic * q)]
                out_threshold_minor = sort_minor[int(n_minor * q)]

                out_threshold_list_major.append(out_threshold_major)
                # out_threshold_list_major_classic.append(out_threshold_major_classic)
                out_threshold_list_minor.append(out_threshold_minor)

            dictionary_major.update({pca.name: out_threshold_list_major})
            # dictionary_major_classic.update({pca.name: out_threshold_list_major_classic})
            dictionary_minor.update({pca.name: out_threshold_list_minor})
        self.dictionary_major = dictionary_major
        # self.dictionary_major_classic = dictionary_major_classic
        self.dictionary_minor = dictionary_minor
    else:

        dictionary_major_corr = {}
        dictionary_minor_corr = {}

        dictionary_major_cosine = {}
        dictionary_minor_cosine = {}

        for e, pca in enumerate(self.PCAs):

            r = len(pca.estimate_fs[pca.estimate_fs < self.minorSVvariance])
            p = len(pca.estimate_fs)

            out_threshold_list_major = []
            out_threshold_list_minor = []

            out_threshold_list_major_cosine = []
            out_threshold_list_minor_cosine = []

            out_threshold_list_major_corr = []
            out_threshold_list_minor_corr = []

            emp_distribution_major_corr = []
            emp_distribution_minor_corr = []

            dotted = training_sets[e].dot(pca.estimate_right_sv.T)
            if isinstance(dotted, pd.DataFrame):
                if experiment == 0:
                    s_major = np.sum(dotted ** 2 / pca.estimate_fs, axis=1)
                else:
                    s_major = np.sum(dotted.iloc[:, :np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                     self.retained_variance[e],
                                                                     side='right') + 1] ** 2 /
                                     pca.estimate_fs[:np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                      self.retained_variance[e], side='right') + 1],
                                     axis=1)
            else:
                if experiment == 0:
                    s_major = np.sum(dotted ** 2 / pca.estimate_fs, axis=1)
                else:
                    s_major = np.sum(dotted[:, :np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                self.retained_variance[e], side='right') + 1] ** 2 /
                                     pca.estimate_fs[:np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                                      self.retained_variance[e], side='right') + 1],
                                     axis=1)

            cosine_together = cosine_similarity(training_sets[e], pca.estimate_right_sv)
            if experiment == 0:
                s_major_cosine = np.sum(cosine_together ** 2 / pca.estimate_fs, axis=1)
            else:
                s_major_cosine = np.sum(
                    cosine_together[:, :np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                        self.retained_variance[e], side='right') + 1] ** 2 /
                    pca.estimate_fs[:np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                     self.retained_variance[e], side='right') + 1],
                    axis=1)

            if len((np.where(np.isclose(pca.estimate_fs, 0)))[0]) != 0:
                if isinstance(dotted, pd.DataFrame):
                    s_minor = np.sum(dotted.iloc[:, p - r: np.where(np.isclose(pca.estimate_fs, 0))[0][
                        0]] ** 2 / pca.estimate_fs[
                                   p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]], axis=1)
                else:
                    s_minor = np.sum(dotted[:, p - r: np.where(np.isclose(pca.estimate_fs, 0))[0][
                        0]] ** 2 / pca.estimate_fs[
                                   p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]], axis=1)
                s_minor_cosine = np.sum(cosine_together[:, p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][
                    0]] ** 2 / pca.estimate_fs[p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]],
                                        axis=1)
            else:
                if isinstance(dotted, pd.DataFrame):
                    s_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / pca.estimate_fs[p - r:], axis=1)
                else:
                    s_minor = np.sum(dotted[:, p - r:] ** 2 / pca.estimate_fs[p - r:], axis=1)

                s_minor_cosine = np.sum(cosine_together[:, p - r:] ** 2 / pca.estimate_fs[p - r:], axis=1)

            for j in range(len(training_sets[e])):
                if isinstance(training_sets[e], pd.DataFrame):
                    y_corr = np.corrcoef(training_sets[e].iloc[j], pca.estimate_right_sv)[0][1:]
                else:
                    y_corr = np.corrcoef(training_sets[e][j], pca.estimate_right_sv)[0][1:]

                if experiment == 0:
                    s_major_corr = np.sum(y_corr ** 2 / pca.estimate_fs, axis=1)
                else:
                    s_major_corr = np.sum(
                        y_corr[:np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                self.retained_variance[e], side='right') + 1] ** 2 /
                        pca.estimate_fs[:np.searchsorted(stable_cumsum(pca.estimate_fs_ratio),
                                                         self.retained_variance[e], side='right') + 1])
                if len((np.where(np.isclose(pca.estimate_fs, 0)))[0]) != 0:
                    s_minor_corr = np.sum(y_corr[p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][
                        0]] ** 2 / pca.estimate_fs[
                                   p - r:np.where(np.isclose(pca.estimate_fs, 0))[0][0]])
                else:
                    s_minor_corr = np.sum(y_corr[p - r:] ** 2 / pca.estimate_fs[p - r:])

                emp_distribution_major_corr.append(s_major_corr)
                emp_distribution_minor_corr.append(s_minor_corr)

            if isinstance(dotted, pd.DataFrame):
                emp_distribution_major = s_major.values
                emp_distribution_minor = s_minor.values
            else:
                emp_distribution_major = s_major
                emp_distribution_minor = s_minor
            emp_distribution_major_cosine = s_major_cosine
            emp_distribution_minor_cosine = s_minor_cosine

            for _, q in enumerate(self.quantils):
                n_major = len(emp_distribution_major)
                n_minor = len(emp_distribution_minor)

                n_major_corr = len(emp_distribution_major_corr)
                n_minor_corr = len(emp_distribution_minor_corr)

                n_major_cosine = len(emp_distribution_major_cosine)
                n_minor_cosine = len(emp_distribution_minor_cosine)

                sort_major = sorted(emp_distribution_major)
                sort_minor = sorted(emp_distribution_minor)

                sort_major_corr = sorted(emp_distribution_major_corr)
                sort_minor_corr = sorted(emp_distribution_minor_corr)

                sort_major_cosine = sorted(emp_distribution_major_cosine)
                sort_minor_cosine = sorted(emp_distribution_minor_cosine)

                out_threshold_major = sort_major[int(n_major * q)]
                out_threshold_minor = sort_minor[int(n_minor * q)]

                out_threshold_major_corr = sort_major_corr[int(n_major_corr * q)]
                out_threshold_minor_corr = sort_minor_corr[int(n_minor_corr * q)]

                out_threshold_major_cosine = sort_major_cosine[int(n_major_cosine * q)]
                out_threshold_minor_cosine = sort_minor_cosine[int(n_minor_cosine * q)]

                out_threshold_list_major.append(out_threshold_major)
                out_threshold_list_minor.append(out_threshold_minor)

                out_threshold_list_major_corr.append(out_threshold_major_corr)
                out_threshold_list_minor_corr.append(out_threshold_minor_corr)

                out_threshold_list_major_cosine.append(out_threshold_major_cosine)
                out_threshold_list_minor_cosine.append(out_threshold_minor_cosine)

            dictionary_major.update({pca.name: out_threshold_list_major})
            dictionary_minor.update({pca.name: out_threshold_list_minor})

            dictionary_major_corr.update({pca.name: out_threshold_list_major_corr})
            dictionary_minor_corr.update({pca.name: out_threshold_list_minor_corr})

            dictionary_major_cosine.update({pca.name: out_threshold_list_major_cosine})
            dictionary_minor_cosine.update({pca.name: out_threshold_list_minor_cosine})

        self.dictionary_major = dictionary_major
        self.dictionary_minor = dictionary_minor

        self.dictionary_major_corr = dictionary_major_corr
        self.dictionary_minor_corr = dictionary_minor_corr

        self.dictionary_major_cosine = dictionary_major_cosine
        self.dictionary_minor_cosine = dictionary_minor_cosine

    return self

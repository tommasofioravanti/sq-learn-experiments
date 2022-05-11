import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def fit_normal(self, training_sets, minor_sv_variance, only_dot_product):

    self.minorSVvariance = minor_sv_variance
    dictionary_major = {}
    dictionary_minor = {}

    if only_dot_product:
        for e, pca in enumerate(self.PCAs):

            r = len(pca.explained_variance_[pca.explained_variance_ < minor_sv_variance])

            p = len(pca.explained_variance_)

            out_threshold_list_major = []
            out_threshold_list_minor = []

            dotted = training_sets[e].dot(pca.components_.T)
            if isinstance(dotted, pd.DataFrame):

                s_major = np.sum(dotted.iloc[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                    axis=1)
            else:
                s_major = np.sum(dotted[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],
                    axis=1)

            if len((np.where(np.isclose(pca.explained_variance_, 0)))[0]) != 0:
                if isinstance(dotted, pd.DataFrame):

                    s_minor = np.sum(dotted.iloc[:, p - r: np.where(np.isclose(pca.explained_variance_, 0))[0][
                        0]] ** 2 / pca.explained_variance_[p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]],
                                     axis=1)
                else:
                    s_minor = np.sum(dotted[:, p - r: np.where(np.isclose(pca.explained_variance_, 0))[0][
                        0]] ** 2 / pca.explained_variance_[
                                   p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]],
                                     axis=1)
            else:
                if isinstance(dotted, pd.DataFrame):
                    s_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / pca.explained_variance_[p - r:], axis=1)
                else:
                    s_minor = np.sum(dotted[:, p - r:] ** 2 / pca.explained_variance_[p - r:], axis=1)

            if isinstance(dotted, pd.DataFrame):
                emp_distribution_major = s_major.values
                emp_distribution_minor = s_minor.values
            else:
                emp_distribution_major = s_major
                emp_distribution_minor = s_minor

            for _, q in enumerate(self.quantils):
                n_major = len(emp_distribution_major)
                n_minor = len(emp_distribution_minor)

                sort_major = sorted(emp_distribution_major)
                sort_minor = sorted(emp_distribution_minor)

                out_threshold_major = sort_major[int(n_major * q)]
                out_threshold_minor = sort_minor[int(n_minor * q)]

                out_threshold_list_major.append(out_threshold_major)
                out_threshold_list_minor.append(out_threshold_minor)

            dictionary_major.update({pca.name: out_threshold_list_major})
            dictionary_minor.update({pca.name: out_threshold_list_minor})
        self.dictionary_major = dictionary_major
        self.dictionary_minor = dictionary_minor
    else:

        dictionary_major_corr = {}
        dictionary_minor_corr = {}

        dictionary_major_cosine = {}
        dictionary_minor_cosine = {}

        for e, pca in enumerate(self.PCAs):

            r = len(pca.explained_variance_[pca.explained_variance_ < minor_sv_variance])

            p = len(pca.explained_variance_)

            out_threshold_list_major = []
            out_threshold_list_minor = []

            out_threshold_list_major_cosine = []
            out_threshold_list_minor_cosine = []

            out_threshold_list_major_corr = []
            out_threshold_list_minor_corr = []

            emp_distribution_major_corr = []
            emp_distribution_minor_corr = []

            dotted = training_sets[e].dot(pca.components_.T)
            if isinstance(dotted, pd.DataFrame):
                s_major = np.sum(dotted.iloc[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_], axis=1)
            else:
                s_major = np.sum(dotted[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_],axis=1)

            cosine_together = cosine_similarity(training_sets[e], pca.components_)
            s_major_cosine = np.sum(cosine_together[:, :pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_], axis=1)

            if len((np.where(np.isclose(pca.explained_variance_, 0)))[0]) != 0:
                if isinstance(dotted, pd.DataFrame):
                    s_minor = np.sum(dotted.iloc[:, p - r: np.where(np.isclose(pca.explained_variance_, 0))[0][0]] ** 2 / pca.explained_variance_[p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]], axis=1)
                else:
                    s_minor = np.sum(dotted[:, p - r: np.where(np.isclose(pca.explained_variance_, 0))[0][0]] ** 2 / pca.explained_variance_[p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]], axis=1)
                s_minor_cosine = np.sum(cosine_together[:, p-r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]]**2 / pca.explained_variance_[p-r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]], axis=1)
            else:
                if isinstance(dotted, pd.DataFrame):
                    s_minor = np.sum(dotted.iloc[:, p - r:] ** 2 / pca.explained_variance_[p - r:], axis=1)
                else:
                    s_minor = np.sum(dotted[:, p - r:] ** 2 / pca.explained_variance_[p - r:], axis=1)

                s_minor_cosine = np.sum(cosine_together[:, p-r:]**2 / pca.explained_variance_[p-r:], axis=1)

            for j in range(len(training_sets[e])):
                if isinstance(training_sets[e], pd.DataFrame):
                    y_corr = np.corrcoef(training_sets[e].iloc[j], pca.components_)[0][1:]
                else:
                    y_corr = np.corrcoef(training_sets[e][j], pca.components_)[0][1:]

                s_major_corr = np.sum(y_corr[:pca.components_retained_] ** 2 / pca.explained_variance_[:pca.components_retained_])
                if len((np.where(np.isclose(pca.explained_variance_, 0)))[0]) != 0:
                    s_minor_corr = np.sum(y_corr[p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]] ** 2 / pca.explained_variance_[p - r:np.where(np.isclose(pca.explained_variance_, 0))[0][0]])
                else:
                    s_minor_corr = np.sum(y_corr[p - r:] ** 2 / pca.explained_variance_[p - r:])

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

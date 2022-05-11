## Wrapper of the model with only dot product and with the other 3 summatory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Models.Fit.QuantumRoutines import *
from Models.Fit.NormalRoutines import *
from Models.Predict.NormalPredict import *
from Models.Predict.QuantumPredict import *


class Model:

    def __init__(self, PCAs, quantils, quantum=False):
        self.PCAs = PCAs
        self.quantils = quantils
        self.quantum = quantum

    def fit(self, training_sets, minor_sv_variance,experiment, only_dot_product=True):
        if self.quantum:

            model = fit_new_quantum(self=self, training_sets=training_sets, minor_sv_variance=minor_sv_variance,
                                only_dot_product=only_dot_product, experiment=experiment)
        else:
            model = fit_normal(self=self, training_sets=training_sets, minor_sv_variance=minor_sv_variance,
                               only_dot_product=only_dot_product)
        return model

    def predict(self, tests, labels, name_negative_class, only_dot_product=True, experiment=0):
        if self.quantum:
            if only_dot_product:
                recall_res, prec_res, acc_res, f1_score_res = only_dot_product_quantum_new(self=self, experiment=experiment,
                                                                                       tests=tests,
                                                                                       labels=labels,
                                                                                       name_neg_class=name_negative_class)
            else:
                recall_res, prec_res, acc_res, f1_score_res = dot_cosine_corr_measure_quantum_new(self=self,
                                                                                              experiment=experiment,
                                                                                              tests=tests,
                                                                                              labels=labels,
                                                                                              name_neg_class=name_negative_class)
        else:
            if only_dot_product:
                recall_res, prec_res, acc_res, f1_score_res = only_dot_prod_normal(self=self, experiment=experiment,
                                                                                   tests=tests,
                                                                                   labels=labels,
                                                                                   name_neg_class=name_negative_class)
            else:
                recall_res, prec_res, acc_res, f1_score_res = dot_cosine_corr_measure_normal(self=self,
                                                                                             experiment=experiment,
                                                                                             tests=tests,
                                                                                             labels=labels,
                                                                                             name_neg_class=name_negative_class)
        return recall_res, prec_res, acc_res, f1_score_res

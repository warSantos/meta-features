from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from classifiers.config import base_estimators, default_params, default_tuning_params
import time
from sklearn.base import clone
import gzip
from sklearn.datasets import dump_svmlight_file
import pickle
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from collections import Counter
from sklearn import exceptions


class TraditionalClassifier(BaseEstimator):

    def __init__(self, args):
        self.args = args
        self.estimator = base_estimators[self.args['name_class']]
        self.params = default_params[self.args['name_class']].copy()
        
        if self.args['name_class'] == 'svm':
            self.params['max_iter'] = args['max_iter']

        self.estimator.set_params(**self.params)

        # Por fold
        self.micro_validation = None
        self.macro_validation = None

        self.grid_time = 0
        self.train_time = 0
        self.test_time = 0

        print(self.estimator)


    def fit(self, X, y=None):

        if self.args['name_class'] == 'knn' and X.shape[0] < 120:
            print("Adjusting KNN Default Tunning parameters")
        
            default_tuning_params[self.args['name_class']][0]['n_neighbors'] = sorted(
                list(set(map(int, np.linspace(1, 0.75*X.shape[0], 5)))))
            print(X.shape)
            print(default_tuning_params[self.args['name_class']])

        #Possibilitando executar o cv para datasets ext pequenos
        counter = Counter(y)
        mininum = min(counter, key=counter.get)
        if counter[mininum] < 10 and self.args['dataset'] not in ['webkb', '20ng', 'acm', 'reut', 'reut90']:
            print(f"Adjusting CV value to {counter[mininum]}")
            self.args['cv'] = counter[mininum]
            print(self.args)

        if self.args['cv'] > 1:
            #t_init = time.time()
            n_jobs = self.args['n_jobs']

            tunning = default_tuning_params[self.args['name_class']]

            scoring = ['f1_micro', 'f1_macro']

            t_init = time.time()

            gs = GridSearchCV(self.estimator, tunning,
                              n_jobs=n_jobs,
                              # refit=False,
                              cv=self.args['cv'],
                              verbose=1,
                              # scoring='f1_macro')
                              scoring=scoring,
                              refit='f1_micro')
            gs.fit(X, y)
            print(gs.best_score_, gs.best_params_)

            self.estimator.set_params(**gs.best_params_)
            self.micro_validation = gs.cv_results_[
                'mean_test_f1_micro'][gs.best_index_]
            self.macro_validation = gs.cv_results_[
                'mean_test_f1_macro'][gs.best_index_]

            self.grid_time = time.time() - t_init

        print(self.estimator)
        self.estimator = clone(self.estimator)

        # fit and predict
        print('Fitting')
        t_init = time.time()
        self.estimator.fit(X, y)
        self.train_time = time.time() - t_init

        # calibrator pro predict proba
        self.calibrator = CalibratedClassifierCV(self.estimator, cv='prefit')
        self.calibrator.fit(X, y)

        return self

    def predict(self, X, y=None):
        print('Predicting')
        t_init = time.time()
        self.y_pred = self.estimator.predict(X)
        self.test_time = time.time() - t_init
        return self.y_pred

    def predict_proba(self, X, y=None):
        return self.calibrator.predict_proba(X)

    def save_proba(self, X, y, f, tipo):
        with gzip.open(self.args['finaloutput']+"proba_"+tipo+"_"+str(f)+".gz", 'w') as filout:
            dump_svmlight_file(X, y, filout, zero_based=False)

    def save_model(self, f):
        pickle.dump(self.estimator, open(
            self.args['finaloutput']+"model_"+str(f), 'wb'))

from genericpath import exists
from itertools import count
import os
from itertools import product
from sys import prefix
import numpy as np
from src.utils.utils import count_to_vector
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2

from src.utils.utils import (
    get_train_test, stratfied_cv, replace_nan_inf, save_mfs, load_df)


class InfoMFS:

    # Get the medium TF-IDF from documents.
    # X_train: count word matrix from train documents.
    # X_test: count word matrix from train documents.
    def tfidf(self, X_train, X_test):

        tf = TfidfTransformer()
        tf.fit(X_train)
        idfs_rep = tf.transform(X_test).toarray()
        docs_idf = np.dot(idfs_rep > 0, tf.idf_)
        adjust = idfs_rep.shape[1] + 1
        norm = adjust - np.count_nonzero(idfs_rep, axis=1)
        return docs_idf / norm

    def tfidf_topnw(self, X_train, X_test, topn=15):

        tf = TfidfTransformer()
        tf.fit(X_train)
        idfs_rep = tf.transform(X_test).toarray()
        idfs_rep.sort(axis=1)
        midf = np.sum(idfs_rep[:, -topn:], axis=1) / topn
        return midf

    def chi2(self, X_train, X_test, y_train):

        chi2_values, _ = chi2(X_train, y_train)
        idfv = np.dot(X_test > 0, chi2_values)
        total = np.count_nonzero(X_test, axis=1)
        return idfv / total

    def chi2_topnw(self, X_train, X_test, y_train, topn=15):

        chi2_values, _ = chi2(X_train, y_train)
        Xc = X_test.copy()
        Xc[Xc > 0] = 1
        cm = Xc * chi2_values
        cm.sort(axis=1)
        mc = np.sum(cm[:, -topn:], axis=1) / topn
        return mc

    def transform(self, train_texts, test_texts, train_classes, params=None):

        cv, X_train = count_to_vector(train_texts)
        X_test = cv.transform(test_texts).toarray()

        mf_idf = self.tfidf_topnw(X_train, X_test, topn=params["topn"])
        mf_chi2 = self.chi2_topnw(
            X_train, X_test, train_classes, topn=params["topn"])

        return mf_idf, mf_chi2

    def build_features(self, dset, params=None, params_prefix=""):

        df = load_df(f"data/datasets/{dset}.csv")

        # For the first cross-val level.
        for fold in np.arange(10):
            # Separate da data in train, test
            train, test = get_train_test(df, fold)
            # List of train mfs.
            train_mfs = []
            # Make new splits to generate train MFs.
            splits = stratfied_cv(train.docs, train.classes, dset=dset, load_splits=False)
            for inner_fold in splits.itertuples():

                inner_train_texts = train.docs.values[inner_fold.train]
                inner_train_classes = train.classes.values[inner_fold.train]
                inner_test_texts = train.docs.values[inner_fold.test]

                inner_idf, inner_chi2 = self.transform(
                    inner_train_texts, inner_test_texts, inner_train_classes, params=params)

                new_mfs = np.vstack([inner_idf, inner_chi2]).T
                train_mfs.append(new_mfs)

            train_mfs = np.vstack(train_mfs)

            # Generating test meta-features.
            outer_idf, outer_chi2 = self.transform(train.docs.values, test.docs.values,
                                               train.classes.values, params=params)
            test_mfs = np.vstack([outer_idf, outer_chi2]).T

            train_mfs = replace_nan_inf(train_mfs)
            test_mfs = replace_nan_inf(test_mfs)

            save_mfs(dset, "info", fold, train_mfs, test_mfs, params_prefix=params_prefix)

    def build(self, datasets=["webkb", "20ng", "reut", "acm"]):
        
        params = {"topn": [15, 30]}

        for dset, topn in product(datasets, params["topn"]):
            iter_params = {
                "topn": topn
            }
            params_prefix = f"topn/{topn}"
            print(f"Dataset: {dset}\n\tParameters: {params_prefix}", end="\r")
            self.build_features(dset, iter_params, params_prefix=params_prefix)

"""
from src.stat_mfs.info import InfoMFS
inf = InfoMFS()
inf.build()
"""

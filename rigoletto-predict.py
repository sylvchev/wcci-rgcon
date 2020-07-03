from warnings import simplefilter

from joblib import Parallel, delayed

import mne
from mne.decoding import CSP

import numpy as np

import pandas as pd

from pyriemann.classification import MDM
from pyriemann.clustering import Potato
from pyriemann.tangentspace import FGDA
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_covariance

from scipy.io import loadmat

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import unique_labels

from tqdm import trange


###############################################################################
# Local function
###############################################################################

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def isPD2(B):
    """Returns true when input is positive-definite, via Cholesky"""
    if np.any(np.linalg.eigvals(B) < 0.):
        return False
    else:
        return True


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): htttps://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD2(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3


def load_data(feat, subj, idx, phase):
    if phase == "train":
        if feat == "signal":
            return ep[subj].get_data()[idx]
        elif feat == "Cov":
            d = loadmat("Matlab/Matlab_db/Training/Cov_Training_All.mat",
                        squeeze_me=True)
        else:
            d = loadmat("Matlab/Matlab_db/Training/" +
                        feat + "_Training_121280.mat",
                        squeeze_me=True)
        X = np.array(np.transpose(d['Mat'+feat][subj, 0], axes=(2, 0, 1)))
        for i in range(80):
            if not isPD2(X[i]):
                X[i] = nearestPD(X[i])
        if feat == 'AEC' or feat == 'ICoh':
            X = np.array([X[i] @ X[i] for i in range(80)])
        return X[idx]
    else:
        # for test subject 9 and 10
        if subj == 8 or subj == 9:
            if feat == "Cov":
                d = loadmat("Matlab/Matlab_db/P09E_P10E/" +
                            "Cov_Testing_P09P10.mat",
                            squeeze_me=True)
            else:
                d = loadmat("Matlab/Matlab_db/P09E_P10E" + feat +
                            "_Testing_P09P10_121240.mat", squeeze_me=True)
            X = np.array(np.transpose(d['Mat'+feat][subj-8], axes=(2, 0, 1)))
            for i in range(len(X)):
                if not isPD2(X[i]):
                    X[i] = nearestPD(X[i])
            return X
        # train for X < 80 else test
        if feat == "signal":
            train_ep = ep[subj].get_data()
            test_ep = eptest[subj].get_data()
            return np.concatenate((train_ep, test_ep), axis=0)[idx]
        elif feat == "Cov":
            d1 = loadmat("Matlab/Matlab_db/Training/Cov_Training_All.mat",
                         squeeze_me=True)
            d2 = loadmat("Matlab/Matlab_db/Testing/Cov_Testing.mat",
                         squeeze_me=True)
        else:
            d1 = loadmat("Matlab/Matlab_db/Training/" +
                         feat + "_Training_121280.mat",
                         squeeze_me=True)
            d2 = loadmat("Matlab/Matlab_db/Testing" +
                         feat + "_Testing_121240.mat",
                         squeeze_me=True)
        X1 = np.array(np.transpose(d1['Mat'+feat][subj, 0], axes=(2, 0, 1)))
        X2 = np.array(np.transpose(d2['Mat'+feat][subj], axes=(2, 0, 1)))
        X = np.concatenate((X1, X2), axis=0)
        for i in range(len(X)):
            if not isPD2(X[i]):
                X[i] = nearestPD(X[i])
        return X[idx]


class FeatConn(TransformerMixin, BaseEstimator):
    """Getting connectivity features from mat files
    """
    def __init__(self, feat="Cov", subj=0, phase="train"):
        self.feat = feat
        self.subj = subj
        self.phase = phase

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return load_data(self.feat, self.subj, X, self.phase)

    def fit_transform(self, X, y=None):
        return self.transform(X)


# PR opened on pyriemann
class FgMDM2(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, metric='riemann', tsupdate=False, n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs
        self.tsupdate = tsupdate

        if isinstance(metric, str):
            self.metric_mean = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self._mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self._fgda = FGDA(metric=self.metric_mean, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y)
        self._mdm.fit(cov, y)
        return self

    def predict(self, X):
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)

    def predict_proba(self, X):
        cov = self._fgda.transform(X)
        return self._mdm.predict_proba(cov)

    def transform(self, X):
        cov = self._fgda.transform(X)
        return self._mdm.transform(cov)


class MDM2(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, metric='riemann', n_jobs=1):
        """Init."""
        # store params for cloning purpose
        self.metric = metric
        self.n_jobs = n_jobs

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.n_jobs == 1:
            self.covmeans_ = [mean_covariance(X[y == l],
                                    metric=self.metric_mean,
                                    sample_weight=sample_weight[y == l])
                              for l in self.classes_]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == l], metric=self.metric_mean,
                                         sample_weight=sample_weight[y == l])
                for l in self.classes_)

        return self

    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.covmeans_)

        if self.n_jobs == 1:
            dist = [distance(covtest, self.covmeans_[m], self.metric_dist)
                    for m in range(Nc)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                covtest, self.covmeans_[m], self.metric_dist)
                for m in range(Nc))

        dist = np.concatenate(dist, axis=1)
        return dist

    def predict(self, covtest):
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        return softmax(-self._predict_distances(X))


###############################################################################
# Remove warnings
###############################################################################

mne.set_log_level('ERROR')

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


###############################################################################
# Within subject prediction
###############################################################################
n_jobs = -1
d_ = loadmat("Matlab/Matlab_db/Training/Coh_Training_121280.mat",
             squeeze_me=True)
y_train_all = []
for i in range(8):
    y_train_all.append(d_['MatCoh'][i, 1])

all_pred = []
for s in trange(8):
    X = np.array([i for i in range(120)])
    y_train = y_train_all[s]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # Patate
    th = 5 # 4
    mat = FeatConn("Coh", s, "test").fit_transform(X)
    pt = Potato(metric='logeuclid', threshold=th).fit(mat)
    atfcoh = (pt.predict(mat) == 1)
    mat = FeatConn("Cov", s, "test").fit_transform(X)
    pt = Potato(metric='logeuclid', threshold=th).fit(mat)
    atfcov = (pt.predict(mat) == 1)
    mat = FeatConn("PLV", s, "test").fit_transform(X)
    pt = Potato(metric='logeuclid', threshold=th).fit(mat)
    atfplv = (pt.predict(mat) == 1)
    atf = np.logical_and(atfcoh, np.logical_and(atfcov, atfplv))
    print ("removed ", len(X)-atf.sum(), "artefacts")
    X_train = X[:80]
    X_train = X_train[atf[:80]]
    # On n'applique pas la patate pour X_test car on doit prédire tout
    X_test = X[80:]
    y_train = y_train[atf[:80]]

    pipelines = {}
    pipelines['fgMDM-Coh'] = make_pipeline(
        FeatConn("Coh", s, "test"),
        FgMDM2(metric='logeuclid', tsupdate=True, n_jobs=n_jobs))
    pipelines['fgMDM-PLV'] = make_pipeline(
        FeatConn("PLV", s, "test"),
        FgMDM2(metric='logeuclid', tsupdate=True, n_jobs=n_jobs))
    pipelines['fgMDM-Cov'] = make_pipeline(
        FeatConn("Cov", s, "test"),
        FgMDM2(metric='logeuclid', tsupdate=True, n_jobs=n_jobs))
    estimators = [('cov', pipelines['fgMDM-Cov']),
                  ('coh', pipelines['fgMDM-Coh']),
                  ('plv', pipelines['fgMDM-PLV'])
                 ]
    final_estimator = RidgeClassifier(class_weight="balanced")
    cvkf = StratifiedKFold(n_splits=5, shuffle=True)
    scl = StackingClassifier(estimators=estimators,
            cv=cvkf, n_jobs=n_jobs, final_estimator=final_estimator,
            stack_method='predict_proba')
    pipelines['Ensemble'] = scl

    pipelines['Ensemble'].fit(X_train, y_train)
    y_pred = pipelines['Ensemble'].predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    for i, yp in enumerate(y_pred):
        res = {"subject name": "P{:02d}".format(s+1),
               "trial index": i+1,
               "prediction": yp}
        all_pred.append(res)
df_pred = pd.DataFrame(all_pred)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('results-within.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
for s in df_pred['subject name'].unique(): 
    df_pred[df_pred['subject name'] == s].to_excel(writer, sheet_name=s, index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()


###############################################################################
# Cross subject prediction
###############################################################################

n_jobs = 1
d_ = loadmat("Matlab/Matlab_db/Training/Coh_Training_121280.mat", squeeze_me=True)
y_train_all = []
for i in range(8):
    y_train_all.append(d_['MatCoh'][i, 1])

cross_subj = []
for s in trange(8):
    X = np.array([i for i in range(80)])
    y_train = y_train_all[s]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    
    # Patate
    th = 3 # 4
    mat = FeatConn("Coh", s, "train").fit_transform(X)
    pt = Potato(metric='logeuclid', threshold=th).fit(mat)
    atfcoh = (pt.predict(mat) == 1)
    mat = FeatConn("PLV", s, "train").fit_transform(X)
    pt = Potato(metric='logeuclid', threshold=th).fit(mat)
    atfplv = (pt.predict(mat) == 1)
    mat = FeatConn("Cov", s, "train").fit_transform(X)
    pt = Potato(metric='logeuclid', threshold=th).fit(mat)
    atfcov = (pt.predict(mat) == 1)
    atf = np.logical_and(atfcoh, np.logical_and(atfcov, atfplv))
    print ("removed ", len(X)-atf.sum(), "artefacts")
    X_train = X[:80]
    X_train = X_train[atf[:80]]
    # On n'applique pas la patate pour X_test car on doit prédire tout
    # X_test = X[60:]
    y_train = y_train[atf[:80]]

    pipelines = {}
    pipelines['fgMDM-Coh'] = make_pipeline(
        FeatConn("Coh", s, "train"),
        FgMDM2(metric='logeuclid', tsupdate=True, n_jobs=n_jobs))
    pipelines['fgMDM-PLV'] = make_pipeline(
        FeatConn("PLV", s, "train"),
        FgMDM2(metric='logeuclid', tsupdate=True, n_jobs=n_jobs))
    pipelines['fgMDM-Cov'] = make_pipeline(
        FeatConn("Cov", s, "train"),
        FgMDM2(metric='logeuclid', tsupdate=True, n_jobs=n_jobs))
    pipelines['CSP-LDA'] = make_pipeline(
        FeatConn("signal", s),
        CSP(n_components=6, reg='ledoit_wolf'), 
        LDA())
    estimators = [('cov', pipelines['fgMDM-Cov']),
                  ('coh', pipelines['fgMDM-Coh']),
                  ('plv', pipelines['fgMDM-PLV'])
                 ]
    final_estimator = LogisticRegression(penalty='elasticnet', l1_ratio=0.1, intercept_scaling=1000., solver='saga') # RidgeClassifier(class_weight="balanced")
    cvkf = StratifiedKFold(n_splits=5, shuffle=True)
    scl = StackingClassifier(estimators=estimators,
            cv=cvkf, n_jobs=n_jobs, final_estimator=final_estimator,
            stack_method='predict_proba')
    pipelines['Ensemble'] = scl

    pipelines['CSP-LDA'].fit(X_train, y_train)
    pipelines['Ensemble'].fit(X_train, y_train)
    cross_subj.append({'X_train': X_train, 'y_train': y_train, 
                       'pipelines': pipelines, 
                       'mean': mean_covariance(mat[atf], metric='riemann', sample_weight=None)})

cov8 = load_data("Cov", 8, range(40), "test")
cov9 = load_data("Cov", 8, range(40), "test")
cross_subj.append({'X_train': np.array([i for i in range(40)]), 
                   'pipelines': pipelines, 
                   'mean': mean_covariance(cov8, metric='riemann', sample_weight=None)})
cross_subj.append({'X_train': np.array([i for i in range(40)]), 
                   'pipelines': pipelines, 
                   'mean': mean_covariance(cov9, metric='riemann', sample_weight=None)})

all_res = []
for target in trange(8,10):
    best = np.argmin(np.array([distance(cross_subj[source]['mean'],
                                        cross_subj[target]['mean'], metric='riemann')
                               for source in range(8) if source != target]))
    yens = cross_subj[best]['pipelines']['Ensemble'].predict(cross_subj[target]['X_train'])
    yens = le.inverse_transform(yens)
    for i, yp in enumerate(yens):
        res = {"subject name": "P{:02d}".format(target+1),
               "trial index": i+1,
               "prediction": yp}
        all_res.append(res)
df_pred_cross = pd.DataFrame(all_res)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('results-cross.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
for s in df_pred_cross['subject name'].unique(): 
    df_pred_cross[df_pred_cross['subject name'] == s].to_excel(writer, sheet_name=s, index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()


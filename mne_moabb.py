import moabb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gzip, pickle

from pyriemann.classification import MDM

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP
from mne.connectivity import spectral_connectivity

from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery, FilterBankLeftRightImagery
from moabb.evaluations import CrossSessionEvaluation
from moabb.pipelines.utils import FilterBank


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

class SpectConn(TransformerMixin, BaseEstimator):
    """Getting connectivity features from epochs files
    """
    def __init__(self, method="coh", mode='multitaper', sfreq=None, 
                 fmin=None, fmax=np.inf, mt_bandwidth=None, 
                 mt_adaptive=False, mt_low_bias=True, cwt_freqs=None,
                 cwt_n_cycles=7, block_size=1000, n_jobs=1, verbose=None):
        self.method = method
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.mt_bandwidth = mt_bandwidth
        self.mt_adaptive = mt_adaptive
        self.mt_low_bias = mt_low_bias
        self.cwt_freqs = cwt_freqs
        self.cwt_n_cycles = cwt_n_cycles
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        if self.sfreq is None:
            self.sfreq = X.info['sfreq']
        n_epochs, n_channels = X.shape[0], X.shape[1]
        epc = np.empty(shape=(n_epochs, n_channels, n_channels))
        for i in range(n_epochs):
            c = spectral_connectivity(X[i,:,:].reshape(1, n_channels, -1),
                                      method=self.method, sfreq=self.sfreq, 
                                      faverage=True, fmin=self.fmin, fmax=self.fmax,
                                      mt_bandwidth=self.mt_bandwidth, 
                                      mt_adaptive=self.mt_adaptive,
                                      mt_low_bias=self.mt_low_bias, 
                                      cwt_freqs=self.cwt_freqs,
                                      cwt_n_cycles=self.cwt_n_cycles, 
                                      block_size=self.block_size, 
                                      n_jobs=self.n_jobs, 
                                      verbose=self.verbose)
            # pour la coherence, la diagonale est nulle. Ajouter l'identite ?
            # c = np.squeeze(c[0]).T + np.squeeze(c[0])
            # c = np.power(np.squeeze(c[0]).T + np.squeeze(c[0]) + np.eye(n_channels), 2)
            c = np.linalg.matrix_power(np.squeeze(c[0]).T + np.squeeze(c[0]) + np.eye(n_channels), 2)
            if not isPD2(c):
                c = nearestPD(c)
            epc[i,:,:] = c
        return epc

    def fit_transform(self, X, y=None):
        return self.transform(X)


pipelines = {}
pipelines['SC+MDM'] = make_pipeline(SpectConn(method='coh', sfreq=250.)
                                    MDM(metric='riemann'))
pipelines['CSP + LDA'] = make_pipeline(CSP(n_components=8),
                                       LDA())

dataset = BNCI2014001()
dataset.subject_list = dataset.subject_list[:2]
datasets = [dataset]
overwrite = True  # set to True if we want to overwrite cached results

# broadband filters
fmin = 8
fmax = 35
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=datasets,
                                    suffix='examples', overwrite=overwrite)
results = evaluation.process(pipelines)
with gzip.open('results-moabb-test.pkz', 'wb') as f:
    pickle.dump(results, f)

sns.catplot(data=results, kind='box', x='subject', y='score', hue='pipeline')


from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, clone
import multiprocessing

def train_model(regressor, X, y):
    regressor_ = clone(regressor)
    regressor_.fit(X,y)
    return regressor_


def use_model(regressor, X):
    return regressor.predict(X)


class MultiDimSkRegressor(BaseEstimator):
    def __init__(self, regressor):
        self.regressor = regressor
        self.num_cores = multiprocessing.cpu_count()

    def fit(self, X, Y):
        self.dim = np.shape(Y)[1]
        self.regressors = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(train_model)(self.regressor, X, Y[:,i]) for i in range(self.dim))
        return self

    def predict(self, X):
        Y = Parallel(n_jobs=self.num_cores, prefer="threads")(
            delayed(use_model)(self.regressors[i], X) for i in range(self.dim))
        return np.transpose(np.vstack(Y))


def sample_baseline(wmeans, nsamples, sig=0.01):
    wouts = np.zeros(shape=(np.shape(wmeans)[0], nsamples, np.shape(wmeans)[1]))
    cSig = np.diag(np.ones(np.shape(wmeans)[1]) * sig)

    for i in range(np.shape(wmeans)[0]):
        wmean = wmeans[i, :]
        for j in range(nsamples):
            wouts[i, j, :] = np.random.multivariate_normal(wmean, cSig)

        wouts[i, 0, :] = wmean

    return wouts




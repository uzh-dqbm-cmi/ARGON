# v2020.05.29
import numpy as np
from sklearn.decomposition import PCA


class myPCA():
    def __init__(self, n_components=-1, isInteger=True, bits=12, opType='int64'):
        self.trained = False
        self.n_components = n_components
        self.Kernels = []
        self.PCA = None
        self.Energy_ratio = []
        self.Energy = []
        self.bits = bits
        self.isInteger = isInteger
        self.opType = opType

    def to_int_(self):
        self.Kernels = np.round(self.Kernels * pow(2, self.bits)).astype(self.opType)

    def PCA_numpy(self, X):
        X = X - np.mean(X.copy(), axis=0)
        X_cov = np.cov(X, rowvar=0)
        eVal, eVect = np.linalg.eig(X_cov)
        idx = np.argsort(eVal)[::-1]
        idx = idx[:self.n_components]
        self.Kernels = np.transpose(eVect[:, idx])
        self.Energy_ratio = eVal / np.sum(eVal)
        self.Energy_ratio = self.Energy_ratio[idx]
        self.Energy = eVal[idx]

    def PCA_sklearn(self, X):
        self.PCA = PCA(n_components=self.n_components)
        self.PCA.fit(X)
        self.Kernels = self.PCA.components_
        self.Energy_ratio = self.PCA.explained_variance_ratio_
        self.Energy = self.PCA.explained_variance_

    def fit(self, X, whichPCA='numpy'):
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        if self.n_components == -1:
            self.n_components = X.shape[-1]
        if whichPCA == 'numpy':
            self.PCA_numpy(X)
        elif whichPCA == 'sklearn':
            self.PCA_sklearn(X)
        else:
            assert (False), "whichPCA only support 'numpy' or 'sklearn'!"
        if self.isInteger == True:
            self.to_int_()
        self.trained = True
        return self

    # transform retains mean
    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        return np.dot(X, np.transpose(self.Kernels))

    def inverse_transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = np.dot(X, self.Kernels)
        if self.isInteger == True:
            X = np.round(X / pow(2, 2 * self.bits)).astype(self.opType)
        return X



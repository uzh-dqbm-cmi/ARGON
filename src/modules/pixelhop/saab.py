# v2020.05.29

# Saab transformation
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA

from src.modules.pixelhop.myPCA import myPCA


class Saab():
    def __init__(self, num_kernels=-1, useDC=True, isInteger=False, bits=8, opType='int64', energyTH=-1, trained=False,
                 Kernels=[], Mean0=[], Bias=[]):
        self.par = None
        self.Kernels = Kernels
        self.Bias = Bias
        self.Mean0 = Mean0
        self.Energy = []
        self.num_kernels = num_kernels
        self.useDC = useDC  # cleaned up all needBias
        # self.needBias = needBias
        #########################################
        # self.needBias = useDC
        #########################################
        self.trained = trained
        self.bits = bits
        self.isInteger = isInteger
        self.opType = opType
        #########################################
        self.energyTH = energyTH
        #########################################

    def remove_mean(self, X, axis):
        feature_mean = np.mean(X, axis=axis, keepdims=True)
        X = X - feature_mean
        return X, feature_mean

    def to_int_(self):
        assert (self.useDC == False), "Integer transformation is only supported when 'useDC=False'!"
        self.Bias = np.round(self.Bias * pow(2, self.bits) + 1).astype(self.opType)
        self.Kernels = np.round(self.Kernels * pow(2, self.bits)).astype(self.opType)
        self.Mean0 = np.round(self.Mean0 * pow(2, self.bits)).astype(self.opType)

    def fit(self, X, whichPCA='sklearn'):
        assert (len(X.shape) == 2), "Input must be a 2D array!"
        X = X.astype('float32')

        print("X.shape")
        print(X.shape)  # (100,64)
        print("#########################")

        self.Bias = np.max(np.linalg.norm(X, axis=1)) * 1 / np.sqrt(X.shape[1])
        if self.useDC == True:
            X, dc = self.remove_mean(X.copy(), axis=1)
        X, self.Mean0 = self.remove_mean(X.copy(), axis=0)
        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]  # 64

        self.pca = myPCA(n_components=self.num_kernels, isInteger=False)
        self.pca.fit(X, whichPCA=whichPCA)

        kernels = self.pca.Kernels
        energy = self.pca.Energy_ratio

        print("kernels.shape:")
        print(kernels.shape)
        print("#########################")

        print("energy.shape:")
        print(energy.shape)
        print("#########################")

        if self.useDC == True:
            largest_ev = np.var(dc * np.sqrt(X.shape[-1]))
            dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1])) / np.sqrt(largest_ev)
            kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)
            energy = np.concatenate((np.array([largest_ev]), self.pca.Energy[:-1]), axis=0)
            energy = energy / np.sum(energy)

        #########################################
        if self.num_kernels == -1 and self.energyTH != -1:
            kernelsAfterTH = []
            energyAfterTH = []
            i = 0
            sum = 0.0
            for ele in energy:
                sum += energy[i]
                kernelsAfterTH.append(kernels[i])
                energyAfterTH.append(energy[i])
                if sum > self.energyTH:
                    break
                i = i + 1
            kernels = kernelsAfterTH
            energy = energyAfterTH
        #########################################

        self.Kernels, self.Energy = kernels, energy

        print("Kernels.shape:")
        print(self.Kernels.shape)
        print("#########################")

        print("Energy.shape:")
        print(self.Energy.shape)
        print("#########################")

        print("Energy:")
        print(self.Energy)
        print("#########################")

        if self.isInteger == True:
            self.to_int_()
        self.trained = True
        return self

    def transform(self, X, addBias=True):
        assert (self.trained == True), "Must call fit first!"
        X = X.astype('float32')
        if self.useDC == True:
            X -= self.Mean0  # ？？？
        if self.useDC == True and addBias == True:
            X += self.Bias
        X = np.matmul(X, np.transpose(self.Kernels))
        if self.useDC == True and addBias == True:
            X[:, 0] -= self.Bias
        if self.isInteger == True:
            X = X.astype(self.opType)
        return X

    def inverse_transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        if (self.useDC == True):
            print("       <Warning> May result larger reconstruction error!")
        X = X.astype('float32')
        if self.useDC == True:
            X[:, 0] += self.Bias
        X = np.matmul(X, self.Kernels)
        if self.useDC == True:
            X -= self.Bias
        if self.useDC == True:
            X += self.Mean0
        if self.isInteger == True:
            X = np.round(X / pow(2, 2 * self.bits)).astype(self.opType)
        return X


if __name__ == "__main__":
    from sklearn import datasets

    print(" > This is a test example: ")
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s" % str(data.shape))
    print(" --> test inv")
    print(" -----> num_kernels=-1, useDC=True")
    X = data.copy()  # (1797,8,8,1)
    X = X.reshape(X.shape[0], -1)[0:100]  # (100,64)
    saab = Saab(num_kernels=32, useDC=True)
    saab.fit(X, whichPCA='numpy')
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    print(np.mean(np.abs(X - Y)))
    # assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, useDC=True")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=True)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    print(np.mean(np.abs(X - Y)))
    # assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, useDC=False")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    assert (np.mean(np.abs(X - Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, useDC=False")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    assert (np.mean(np.abs(X - Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, useDC=False, isInteger=True")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False, isInteger=True, bits=16)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    assert (np.mean(np.abs(X - Y)) < 1e-5), "invSaab error!"

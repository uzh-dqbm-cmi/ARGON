# 2020.05.29
# A generalized version of channel wise Saab
# Current code accepts <np.array> shape(..., D) as input
#
# Depth goal may not achieved is no nodes's energy is larger than energy threshold or too few SaabArgs/shrinkArgs, (warning generates)
#
import numpy as np
from sklearn.decomposition import PCA

from src.modules.pixelhop.saab import Saab


class cwSaab():
    def __init__(self, depth=1, energyTH=0.01, SaabArgs=None, shrinkArgs=None, concatArg=None, splitMode=2,
                 cwHop1=False, kernelRetainArg=None):
        self.par = {}
        assert (depth > 0), "'depth' must > 0!"
        self.depth = (int)(depth)
        self.energyTH = energyTH
        assert (SaabArgs != None), "Need parameter 'SaabArgs'!"
        self.SaabArgs = SaabArgs
        assert (shrinkArgs != None), "Need parameter 'shrinkArgs'!"
        self.shrinkArgs = shrinkArgs
        assert (concatArg != None), "Need parameter 'concatArg'!"
        self.concatArg = concatArg
        self.Energy = []
        self.splitidx = []
        self.trained = False
        self.split = False
        self.splitMode = splitMode
        self.cwHop1 = cwHop1
        self.kernelRetainArg = kernelRetainArg
        if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
            self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
            print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, actual depth: %s" % (
            str(depth), str(self.depth)))

    def judge_abs_energy(self, eng):
        return (eng > self.energyTH)

    def judge_energy_ratio(self, X, R1, layer):
        X = self.shrinkArgs[layer]['func'](X, self.shrinkArgs[layer])
        print("X = self.shrinkArgs[layer]['func'](X, self.shrinkArgs[layer])")
        print(X.shape)
        print("##########################")
        X = X.reshape(-1, X.shape[-1]) - np.mean(X.reshape(-1, X.shape[-1]), axis=1, keepdims=True)
        print("X = X.reshape(-1, X.shape[-1]) - np.mean(X.reshape(-1, X.shape[-1]), axis=1, keepdims=True)")
        print(X.shape)
        print("##########################")
        pca = PCA(n_components=1, svd_solver='auto').fit(X)
        R2 = pca.explained_variance_ratio_[0]
        return (R1 / R2 >= self.energyTH)

    def judge_mean_abs_value(self, X, layer):
        X = self.shrinkArgs[layer]['func'](X, self.shrinkArgs[layer])
        tmp = np.moveaxis(X, -1, 0)[0]
        R1 = np.abs(tmp.reshape(-1, 1))
        R2 = np.mean(np.abs(X.reshape(-1, X.shape[-1])), axis=-1, keepdims=True)
        R = np.mean(R2 / R1)
        return (R > self.energyTH)

    def split_(self, X, eng, layer):
        if self.splitMode == 0:
            return self.judge_abs_energy(eng)
        elif self.splitMode == 1:
            return self.judge_mean_abs_value(X, layer)
        elif self.splitMode == 2:
            return self.judge_energy_ratio(X, eng, layer)
        else:
            raise ValueError("Unsupport split mode! Supported: 0, 1, 2")

    def SaabTransform(self, X, saab, train, layer):
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        #################################
        numKernels = -1
        if self.kernelRetainArg != None:
            numKernels = self.kernelRetainArg['Layer' + str(layer)]
            if self.kernelRetainArg['Layer' + str(layer)] != -1:
                S[-1] = self.kernelRetainArg['Layer' + str(layer)]
        else:
            numKernels = SaabArg['num_AC_kernels']
            if SaabArg['num_AC_kernels'] != -1:
                S[-1] = SaabArg['num_AC_kernels']

        if train == True:
            isInteger, bits, opType, whichPCA = False, 8, 'int32', 'numpy'
            if 'isInteger' in SaabArg.keys():
                isInteger = SaabArg['isInteger']
            if 'bits' in SaabArg.keys():
                bits = SaabArg['bits']
            if 'opType' in SaabArg.keys():
                opType = SaabArg['opType']
            if 'whichPCA' in SaabArg.keys():
                whichPCA = SaabArg['opType']

            saab = Saab(num_kernels=numKernels, useDC=SaabArg['useDC'], isInteger=isInteger, bits=bits, opType=opType)
            saab.fit(X, whichPCA=whichPCA)
        #################################

        transformed = saab.transform(X).reshape(S)
        return saab, transformed

    def cwSaab_1_layer(self, X, train):
        print("cwSaab_1_layer X.shape")
        print(X.shape)  # (N,S,S,K)
        print("############")
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer' + str(0)]
        transformed, eng = [], []

        if train == True:
            saab, transformed = self.SaabTransform(X, saab=None, train=True, layer=0)
            saab_cur.append(saab)
            eng.append(saab.Energy)
        else:
            _, transformed = self.SaabTransform(X, saab=saab_cur[0], train=False, layer=0)

        if train == True:
            self.par['Layer' + str(0)] = saab_cur
            self.Energy.append(np.concatenate(eng, axis=0))
        return transformed

    def cwSaab_1_layer_cw(self, X, train):
        print("cwSaab_1_layer_cw X.shape")
        print(X.shape)  # (N,S,S,K)
        print("############")
        S = list(X.shape)
        # print("S")
        # print(S)
        S[-1] = 1
        print("S")
        print(S)  # (N,S,S,1)
        X = np.moveaxis(X, -1, 0)  # (K,N,S,S)
        # print("X = np.moveaxis(X, -1, 0)")
        # print(X.shape)
        # print("############")
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer' + str(0)]
        transformed, eng = [], []
        print(" X.shape[0]: %s" % str(X.shape[0]))
        for i in range(X.shape[0]):  # 0~K-1
            X_tmp = X[i].reshape(S)  # (N,S,S)->(N,S,S,1)
            # print("cwSaab_1_layer_cw X_tmp.shape")
            # print(X_tmp.shape)
            # print("############")
            if train == True:
                print("X_tmp tmp_transformed")
                print(X_tmp.shape)
                print("############")
                saab, tmp_transformed = self.SaabTransform(X_tmp, saab=None, train=True, layer=0)
                saab_cur.append(saab)  # will have K saab objects in the list
                eng.append(saab.Energy)  # will have K saab.Energy in the list
                print("tmp_transformed")
                print(tmp_transformed.shape)
                print("############")

            else:
                if len(saab_cur) == i:
                    break
                _, tmp_transformed = self.SaabTransform(X_tmp, saab=saab_cur[i], train=False, layer=0)

            transformed.append(tmp_transformed)  # concatenate all K transformed output together

        # print("saab_cur")
        # print(saab_cur)
        # print("#########################")

        if train == True:
            self.par['Layer' + str(0)] = saab_cur
            self.Energy.append(np.concatenate(eng, axis=0))  # concatenate with another layer?
        return np.concatenate(transformed, axis=-1)

    def cwSaab_n_layer(self, X, train, layer):
        output, eng_cur, ct, pidx = [], [], -1, 0
        S = list(X.shape)
        S[-1] = 1  # (N,S,S,1)
        X = np.moveaxis(X, -1, 0)  # (K_i,N,S,S)
        saab_prev = self.par['Layer' + str(layer - 1)]
        if train == True:
            saab_cur, splitidx = [], []
        else:
            saab_cur = self.par['Layer' + str(layer)]
        for i in range(len(saab_prev)):
            for j in range(saab_prev[i].Energy.shape[0]):  # K_i-1
                ct += 1
                X_tmp = X[ct].reshape(S)  # (N,S,S,1)
                if train == True:
                    tidx = self.split_(X_tmp, saab_prev[i].Energy[j], layer)
                    # we look at the previous layer energy to see whether we should transform the current layer
                    print("tidx got")
                    print(tidx)
                    splitidx.append(tidx)  # a vector with 0 and 1
                else:
                    tidx = self.splitidx[layer - 1][ct]

                if tidx == False:
                    continue

                # we need to transform the current layer
                self.split = True
                if train == True:
                    saab, out_tmp = self.SaabTransform(X_tmp, saab=None, train=True, layer=layer)
                    saab.Energy *= saab_prev[i].Energy[j]
                    saab_cur.append(saab)  # will have less or equal to K_i saab objects in the list
                    eng_cur.append(saab.Energy)  # will have less or equal to K_i saab.Energy in the list
                else:
                    _, out_tmp = self.SaabTransform(X_tmp, saab=saab_cur[pidx], train=False, layer=layer)
                    pidx += 1

                output.append(out_tmp)  # will contain less or equal to K_i transformed outputs in the list

        if self.split == True:
            output = np.concatenate(output, axis=-1)  # concatenate all the transformed outputs into one vector
            print("shape of output")
            print(output.shape)
            print("###################")
            if train == True:
                self.splitidx.append(splitidx)
                self.par['Layer' + str(layer)] = saab_cur
                self.Energy.append(np.concatenate(eng_cur, axis=0))
                print("self.Energy")
                print(self.Energy)
                print("###################")
                print("self.splitidx")
                print(self.splitidx)
                print("###################")

        # print("self.par")
        # print(self.par)
        # print("###################")
        return output

    def fit(self, X):
        output = []
        if self.cwHop1 == False:
            X = self.cwSaab_1_layer(X, train=True)
        else:
            X = self.cwSaab_1_layer_cw(X, train=True)
        output.append(X)
        for i in range(1, self.depth):
            print("!!!")
            print("self.splitMode %s" % str(self.splitMode))
            print("self.depth %s" % str(self.depth))
            X = self.cwSaab_n_layer(X, train=True, layer=i)
            if self.split == False:  # we may call self.cwSaab_n_layer multiple times
                self.depth = i  # update to the actual depth value
                print("       <WARNING> Cannot futher split, actual depth: %s" % str(i))
                break
            output.append(X)
            self.split = False
        self.trained = True
        return self

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        output = []
        if self.cwHop1 == False:
            X = self.cwSaab_1_layer(X, train=False)
        else:
            X = self.cwSaab_1_layer_cw(X, train=False)
        output.append(X)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=False, layer=i)
            output.append(X)
        assert ('func' in self.concatArg.keys()), "'concatArg' must have key 'func'!"
        output = self.concatArg['func'](output, self.concatArg)
        return output

    def inv_SaabTransform(self, X, saab, inv_shrinkArg):
        assert ('func' in inv_shrinkArg.keys()), "'inv_shrinkArg' must contain key 'func'!"
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        X = saab.inverse_transform(X)
        S[-1] = np.array(X.shape)[-1]
        X = X.reshape(S)
        X = inv_shrinkArg['func'](X, inv_shrinkArg)
        return X

    def inverse_transform(self, X, inv_concatArg, inv_shrinkArgs):
        assert (self.trained == True), "Must call fit first!"
        assert ('func' in inv_concatArg.keys()), "'inv_concatArg' must contain key 'func'!"
        X = inv_concatArg['func'](X, inv_concatArg)
        tmp = np.moveaxis(X[self.depth - 1], -1, 0)
        for i in range(self.depth - 1, -1, -1):
            res, ct = [], 0
            for j in range(len(self.par['Layer' + str(i)])):
                num_kernel = self.par['Layer' + str(i)][j].Energy.shape[0]
                res.append(self.inv_SaabTransform(np.moveaxis(tmp[ct:ct + num_kernel], 0, -1),
                                                  saab=self.par['Layer' + str(i)][j],
                                                  inv_shrinkArg=inv_shrinkArgs[i]))
                ct += num_kernel
            res = np.concatenate(res, axis=-1)
            if i > 0:
                res = np.moveaxis(res, -1, 0)
                tmp = np.moveaxis(X[i - 1], -1, 0)
                ct = 0
                for j in range(tmp.shape[0]):
                    if self.splitidx[i - 1][j] == True:
                        tmp[j] = res[ct]
                        ct += 1
        return res


if __name__ == "__main__":
    # example useage
    from sklearn import datasets
    from skimage.util import view_as_windows


    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = view_as_windows(X, (1, win, win, 1), (1, win, win, 1))
        print("X = view_as_windows(X, (1,win,win,1), (1,win,win,1))")
        print(X.shape)
        print("##########################")
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)


    def invShrink(X, invshrinkArg):
        win = invshrinkArg['win']
        S = X.shape
        X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
        X = np.moveaxis(X, 5, 2)
        X = np.moveaxis(X, 6, 4)
        return X.reshape(S[0], win * S[1], win * S[2], -1)


    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X


    # read data
    import cv2

    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s" % str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'useDC': False, 'batch': None},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None}]
    shrinkArgs = [{'func': Shrink, 'win': 2},
                  {'func': Shrink, 'win': 2},
                  {'func': Shrink, 'win': 2}]
    inv_shrinkArgs = [{'func': invShrink, 'win': 2},
                      {'func': invShrink, 'win': 2},
                      {'func': invShrink, 'win': 2}]
    concatArg = {'func': Concat}
    inv_concatArg = {'func': Concat}

    kernelRetainArg = {'Layer0': -1, 'Layer1': -1, 'Layer2': -1}

    print(" --> test inv")
    print(" -----> depth=1")
    cwsaab = cwSaab(depth=1, energyTH=0.1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg,
                    kernelRetainArg=kernelRetainArg)  # depth=1
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X - Y)) < 1e-5), "invcwSaab error!"
    print(" -----> depth=2")
    cwsaab = cwSaab(depth=2, energyTH=0.5, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg, splitMode=0,
                    cwHop1=True)
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X - Y)) < 1), "invcwSaab error!"
    print(output[0].shape, output[1].shape)
    print("------- DONE -------\n")
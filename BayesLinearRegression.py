#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/11 16:44
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : BayesLinearRegression.py
# @Software: PyCharm Community Edition
import numpy as np
from math import pi, e
import matplotlib.pyplot as plt
import random
import math, copy
# from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


class BayesLR(object):
    def __init__(self):
        self.__basis_type = None
        self.__order = None
        self.__input_x = None
        self.__input_y = None
        self.__alpha = None
        self.__beta = None
        self.__m_n = None
        self.__s_n = None

    @property
    def input_x(self):
        return self.__input_x

    @input_x.setter
    def input_x(self, value):
        if not isinstance(value, list):
            raise ValueError('input_x must be a list or numpy_array')
        self.__input_x = value

    @property
    def input_y(self):
        return self.__input_y

    @input_y.setter
    def input_y(self, value):
        if not isinstance(value, list):
            raise ValueError('input_x must be a list or numpy_array')
        self.__input_y = value

    def basis_function(self, basis_type, order):
        if not isinstance(basis_type, str):
            raise ValueError('basis type must be a list')
        self.__basis_type = basis_type
        self.__order = order
        if self.__basis_type == 'polynomial':
            # order = [ord for ord in range(order)]
            fi = np.zeros([len(self.__input_x), order])
            for i in range(len(self.__input_x)):
                for j in range(order):
                    fi[i, j] = pow(self.__input_x[i], j)
            fi = np.mat(fi)
            fi_fi = fi.T * fi
            eigenvalue, eigenvector = np.linalg.eig(fi_fi)
            return eigenvalue, fi_fi, fi

        elif self.__basis_type == 'gauss':
            pass
        elif self.__basis_type == 'sigmoid':
            pass
        elif self.__basis_type == 'fourier':
            pass
        else:
            raise ValueError('basis type must be: polynomial, gauss, sigmoid or fourier')

    def fit(self, test_x, test_y, alpha, beta, basis_type, order):
        self.__input_x = test_x
        self.__input_y = test_y
        eigenvalue, fi_fi, fi = self.basis_function(basis_type, order)
        eigenvalue = np.mat(eigenvalue)
        fi_fi = np.mat(fi_fi)
        alpha_estimate = 0
        beta_estimate = 0
        m_n = 0
        s_n = 0
        while abs(alpha - alpha_estimate) > 0.001 or abs(beta - beta_estimate) > 0.001:
            s_n = (alpha * np.eye(fi_fi.shape[0], fi_fi.shape[1]) + beta * fi_fi).I
            m_n = beta * s_n * fi.T * np.mat(self.__input_y).T
            eigenvalue_lamda = beta * eigenvalue
            gama = np.mat([float(lamd_i) / (lamd_i + alpha) for lamd_i in np.asarray(eigenvalue_lamda)[0]]).sum()
            alpha_estimate = (float(gama) / (m_n.T * m_n))[0, 0]

            d = np.mat(self.__input_y).T - fi * m_n
            d[:, :] = map(lambda d_: np.power(d_, 2), d[:, :])
            beta_estimate = (len(self.__input_x) - gama) * pow(d.sum(), -1)
            alpha = copy.copy(alpha_estimate)
            beta = copy.copy(beta_estimate)

        self.__alpha = alpha_estimate
        self.__beta = beta_estimate
        self.__m_n = m_n
        self.__s_n = s_n
        print 'fit done!'

    def predict(self, test):
        mean = self.__m_n.T * np.mat([pow(test, i) for i in range(self.__order)]).T
        # print mean
        mean = mean[0, 0]
        # variance = (1.0 / self.__beta) + np.mat([pow(test, i) for i in range(self.__order)]) * self.__s_n *\
        #                                np.mat([pow(test, i) for i in range(self.__order)]).T
        # variance = variance[0, 0]
        # # return pow(2 * pi * variance, -0.5) * pow(e, -(pow((test - mean), 2)/(2*variance)))
        # return pow(2 * pi * variance, -0.5) * math.exp(-(pow((test - mean), 2) / (2 * variance)))
        return mean

    def evaluate(self, test_x, target_y):
        if not isinstance(test_x, list):
            raise ValueError('input_x must be a list')
        if not isinstance(target_y, list):
            raise ValueError('target must be a list')
        if len(test_x) != len(target_y):
            raise ValueError('len of input_x not equal to len of target!')
        y_pred = [self.predict(test_x[i]) for i in range(len(test_x))]
        loss = np.sqrt(mean_squared_error(target_y, y_pred))
        return loss

if __name__ == '__main__':

    x = []
    y = []
    f = open('data.txt').readlines()
    for i in f:
        x.append(float(i.split('\t')[0])), y.append(float(i.split('\t')[1]))

    cc = list(zip(x, y))
    # j = random.sample(cc, 150)
    j = sorted(cc)
    x, y = zip(*j)
    x = list(x)
    y = list(y)
    bayeslr = BayesLR()
    bayeslr.fit(x, y, alpha=0.1, beta=9, basis_type='polynomial', order=7)
    y_pred = [bayeslr.predict(i) for i in x]
    loss = bayeslr.evaluate(x, y)
    print loss

    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    reg = linear_model.LinearRegression()
    # reg = linear_model.ridge_regression(alpha=0.5)
    reg.fit(x, y)
    y_predction = reg.predict(x)
    print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(x, y_predction)))
    ax1 = plt.figure()
    ax1.add_subplot(1, 1, 1)
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.plot(x, y_predction)
    plt.show()









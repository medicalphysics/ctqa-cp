# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:09:04 2014

@author: erlean

Logistic regression borrowed from Marcel Marcelcaraciolo
https://github.com/marcelcaraciolo
"""

#=====================================================
import numpy as np
#from scipy.optimize import fmin_bfgs
from logit import simple_logistic_regression as s_logit


def logit(mX, vBeta):
    return ((np.exp(np.dot(mX, vBeta))/(1.0 + np.exp(np.dot(mX, vBeta)))))


def logLikelihoodLogit(vBeta, mX, vY):
    return (-(np.sum(vY*np.log(logit(mX, vBeta))
            + (1-vY)*(np.log(1-logit(mX, vBeta))))))


def logLikelihoodLogitVerbose(vBeta, mX, vY):
    return (-(np.sum(vY*(np.dot(mX, vBeta)
            - np.log((1.0 + np.exp(np.dot(mX, vBeta)))))
            + (1-vY) * (-np.log((1.0 + np.exp(np.dot(mX, vBeta))))))))


# gradient function
def likelihoodScore(vBeta, mX, vY):
    return(np.dot(mX.T, (logit(mX, vBeta) - vY)))


# Modelfitter
def fitLowContrastScore(PC, d):
    vY = 2. * PC - 1.
    mX = np.log(d)
    try:
        beta, jaq, ll = s_logit(mX, vY, CONV_THRESH=1.e-4)
    except np.linalg.linalg.LinAlgError:
        return False, 0., np.array([]), np.array([])

    if beta[1] < 0.0:
        return False, 0., np.array([]), np.array([])

    lam = np.exp(beta[0]/(-beta[1]))

    dm = np.linspace(1, 15, 50)
    model = np.log(dm.reshape((-1, 1)))
    intm = np.ones(model.shape[0]).reshape(model.shape[0], 1)
    model = np.concatenate((intm, model), axis=1)

    PCm = (logit(model, beta) + 1.) / 2.
    return True, lam, dm, PCm

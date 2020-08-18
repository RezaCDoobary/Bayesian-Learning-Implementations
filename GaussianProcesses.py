#######################################################
#
# Python script responsible for implemented the GPs, 
# particularly, the regression as well as the 
# classification cases.
#
#######################################################

import numpy as np
import pandas as pd
from scipy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
        
class GuassianProcessRegression:
    def __init__(self, kernel, noise = 1e-10):
        self.kernel = kernel
        self.noise = noise
        
    def _nan_checker(self, covariance_matrix):
        """
        A null checker function that is required when taking the square root of the diagonal
        components of the variance matrix. This is required for numercal stability reason, as
        it is possible that we have a very small negative result which we force to be zero.

        TO DO: this needs to be removed.
        """

        diag_part = np.diag(covariance_matrix)
        diag_part_copy = diag_part.copy() ### very very very tiny negative numbers which are essentially zero.
        diag_part_copy[diag_part < 0] = 0
        diag_part = diag_part_copy
        std = np.sqrt(diag_part)
        return std
        
    def _posterior(self, new_point, return_cov = False, return_std = False):
        K_XX = self.kernel.covariance(self.X,self.X) + self.noise*np.eye(len(self.X))
        K_sX = self.kernel.covariance(new_point, self.X)
        K_ss = self.kernel.covariance(new_point, new_point)
        K_XX_inv = inv(K_XX)

        ksxkxxinv = K_sX.dot(K_XX_inv)
        mu_s = ksxkxxinv.dot(self.y) + self.mu_y

        cov_s = K_ss - ksxkxxinv.dot(K_sX.T)
        if return_std:
            std = self._nan_checker(cov_s)
            return mu_s , std
        elif return_cov:
            return mu_s, cov_s
        else:
            return mu_s
        
        
    def fit(self, X, y, config = None):
        """
        Fitting function for the GP regression - this simply fits the hyperparameters.
        Possible config object used if we want to use a slightly different optimisation methodology.
        """
        self.X = X.copy()
        self.y = y.copy()
        mu_y = self.y.mean()
        self.y-=mu_y
        self.mu_y = mu_y
        if not config:
            self.kernel.fit(self.X, self.y)
        else:
            assert(isinstance(config, configs))
            self.kernel.fit(self.X, self.Y, config.init, config, bounds, config, method)
    
    def predict(self, X, return_cov = False, return_std = False):
        res = self._posterior(X, return_cov, return_std)
        return res
    
class configs:
    def __init__(self, init, bounds, method):
        self.init = init
        self.bounds = bounds
        self.method = method

### sigmoid for logisitic regression - should be moved to a different file at some point.
class Sigmoid:
    def __init__(self):
        pass
    
    def prob(self, z):
        return 1/(1+np.exp(-z))
    
    def first_derivative_condtional(self, class_, f_):
        # class_ distributition, conditioned on f_
        return (class_ + 1)/2 - self.prob(f_)
    
    def second_dervative_conditional(self, class_, f_):
        # class_ distributition, conditioned on f_
        return -self.prob(f_)*(1- self.prob(f_))


class GaussianClassification:
    def __init__(self, kernel, class_prob):
        self.kernel = kernel
        self.class_prob = class_prob
        
    def _laplace_approximation(self, f_init, max_iter, tol):
        """
        This implements the laplace approximation exactly as Rasmussen and Williams - Gaussian Processes for Machine Learning book.
        The laplace approximation needs to applied to: P(y|X,c)~ P(gaussian_latent_output|  data, classification).
        """
        prev_logq = None
        f = f_init

        for i in range(0,max_iter):
            
            W = np.diag(-self.class_prob.second_dervative_conditional(self.y, f))
            B = np.eye(len(self.K_XX)) + np.matmul(np.matmul(np.sqrt(W), self.K_XX), np.sqrt(W))
            L = cholesky(B)
            b = np.matmul(W, f) + self.class_prob.first_derivative_condtional(self.y, f)

            B_inv = inv(B)
            a = b - np.matmul(np.matmul(np.matmul(np.matmul(np.sqrt(W),B_inv), np.sqrt(W)),self.K_XX),b)
            f = np.matmul(self.K_XX, a)

            logq = -np.matmul(a,f) + np.sum(-np.log(1+np.exp(-self.y*f))) - np.sum(np.log(np.diag(L)))
            if prev_logq is None:
                prev_logq = logq
            else:
                if abs(prev_logq - logq) < tol:
                    break
            prev_logq = logq
            
        return f, logq

    
    def fit(self, X, y, max_iter = 1000, tol = 10e-6):
        """
        The fitting function simply applies the laplace approximation to P(y|X,c)~ P(gaussian_latent_output|  data, classification)
        """
        self.X = X.copy()
        self.y = y.copy()
        
        self.K_XX = self.kernel.covariance(X, X)
        f = np.array([0]*len(X))
        f, logq = self._laplace_approximation(f, max_iter, tol)
            
        self.logq = logq
        self.f = f
    
    def predict(self, new_data): 

        W = np.diag(self.class_prob.prob(self.f)*(1- self.class_prob.prob(self.f)))
        B = np.eye(len(self.f)) + np.matmul(np.matmul(np.sqrt(W), self.K_XX), np.sqrt(W))
        L = cholesky(B)

        K_sX = self.kernel.covariance(new_data, self.X)
        
        K_ss  = np.diag(np.diag(self.kernel.covariance(new_data,new_data)))
        
        v = np.matmul(K_sX,np.matmul(inv(L),np.sqrt(W)))
        
        
        var = np.diag(np.diag(K_ss - np.matmul(v,v.T)))
        y_mean = np.matmul(K_sX, self.class_prob.first_derivative_condtional(self.y, self.f))
        return y_mean, var
    
    def classify(self, y_mean, threshold = 0.5):
        return np.array(self.class_prob.prob(y_mean) > threshold,int)
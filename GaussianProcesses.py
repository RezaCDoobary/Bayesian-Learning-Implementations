import numpy as np
import pandas as pd
from scipy.linalg import inv

from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import warnings

class kernel:
    def __init__(self):
        pass
    
    def covariance(self, X1, X2):
        pass
    
    def log_prob(self, X, Y, noise):
        pass
    
    def log_prob_core(self, X, Y, noise, naive_implementation = False):
        if not naive_implementation:
            K = self.covariance(X,X) + noise * np.eye(len(X))

            L = cholesky(K)

            return np.sum(np.log(np.diagonal(L))) + \
                   0.5 * Y.T.dot(lstsq(L.T, lstsq(L, Y)[0])[0]) + \
                   0.5 * len(X) * np.log(2*np.pi)
        else:
            K = self.covariance(X,X) + noise * np.eye(len(X))
            #print(det(K))
            return 0.5 * np.log(det(K)) + \
                   0.5 * Y.T.dot(inv(K).dot(Y)) + \
                   0.5 * len(X) * np.log(2*np.pi)
    
    def fit(self, X, Y, noise):
        pass
    
    def reset_parameters(self):
        pass
    
    
class SquareExponential(kernel):
    def __init__(self, length_scale = 1.0, deviation = 1.0):
        self.length_scale = length_scale
        self.deviation = deviation
        self.printed = False
        
    def covariance(self, X1, X2):     
        diff = (X1[:,np.newaxis] - X2)
        square_distance_matrix = np.sum(diff**2,2)
        
        return self.deviation**2 * np.exp(-0.5 / self.length_scale**2 * square_distance_matrix)
    
    def log_prob(self, X, Y, noise, naive_implementation = False, length_scale = None, deviation = None):
        if isinstance(length_scale, float):
            self.length_scale = length_scale
        if isinstance(deviation, float):
            self.deviation = deviation
        
        return self.log_prob_core(X,Y,noise, naive_implementation)
    
    def fit(self, X, Y, noise=1e-10, init = [1,1], bounds = ((1e-5, None), (1e-5, None)), method = 'L-BFGS-B'):

        def to_min(parameter):
            try:
                return self.log_prob(X,Y,noise,naive_implementation = False,\
                                     length_scale = parameter[0],deviation = parameter[1])
            except Exception as e:
                if not self.printed:
                    print('Using unstable log_prob due to the following issue:',e)
                    self.printed = True
                else:
                    pass
                return self.log_prob(X,Y,noise,naive_implementation = True,\
                                     length_scale = parameter[0],deviation = parameter[1])
        res = minimize(to_min, init, bounds=bounds,method=method)
        
    def reset_parameters(self, length_scale, deviation):
        self.length_scale = length_scale
        self.deviation = deviation
        
    def __str__(self):
        outstring = ''
        outstring+='Type : \n'
        outstring+='\t Square Exponential \n'
        outstring+='Parameters : \n'
        outstring+='\t length_scale : '+str(self.length_scale)+'\n'
        outstring+='\t deviation : '+str(self.deviation)
        return outstring
    
    
class RationalQuadraticKernel(kernel):
    def __init__(self, length_scale = 1.0, deviation = 1.0, relative_scaling = 1.0):
        self.length_scale = length_scale
        self.deviation = deviation
        self.relative_scaling = relative_scaling
        self.printed = False
        
    def covariance(self, X1, X2):
        diff = (X1[:,np.newaxis] - X2)
        square_distance_matrix = np.sum(diff**2,2)
        square_distance_matrix = 1 + square_distance_matrix/(2*self.relative_scaling*self.length_scale*self.length_scale)
        return self.deviation**2 * square_distance_matrix**(-self.relative_scaling)
    
    

    
    def log_prob(self, X, Y, noise,naive_implementation = False, length_scale = None, deviation = None, relative_scaling = None):
        if isinstance(length_scale, float):
            self.length_scale = length_scale
        if isinstance(deviation, float):
            self.deviation = deviation
        if isinstance(relative_scaling, float):
            self.relative_scaling = relative_scaling
            
        return self.log_prob_core(X,Y,noise, naive_implementation)
        
    def fit(self, X, Y, noise=1e-10, init = [1,1,1], bounds = ((1e-5, None), (1e-5, None),(1e-5, None)), method = 'L-BFGS-B'):
        def to_min(parameter):
            try:
                return self.log_prob(X,Y,noise,naive_implementation = False,length_scale = parameter[0],deviation = parameter[1], \
                                 relative_scaling = parameter[2])
            except Exception as e:
                if not self.printed:
                    print('Using unstable log_prob due to the following issue:',e)
                    self.printed = True
                else:
                    pass
                return self.log_prob(X,Y,noise,naive_implementation = True,length_scale = parameter[0],deviation = parameter[1], \
                                 relative_scaling = parameter[2])
        res = minimize(to_min, init, bounds=bounds,method=method)
        
    def reset_parameters(self, length_scale, deviation, relative_scaling):
        self.length_scale = length_scale
        self.deviation = deviation
        self.relative_scaling = relative_scaling
        
    def __str__(self):
        outstring = ''
        outstring+='Type : \n'
        outstring+='\t Rational Quadratic Kernel \n'
        outstring+='Parameters : \n'
        outstring+='\t length_scale : '+str(self.length_scale)+'\n'
        outstring+='\t deviation : '+str(self.deviation)+'\n'
        outstring+='\t relative_scaling : '+str(self.relative_scaling)
        return outstring
        
class PeriodicKernel(kernel):
    """
    Wierd error coming from positive definite covariance....already seen in:
    https://stackoverflow.com/questions/55103221/cholesky-decomposition-positive-semidefinite-matrix
    """
    def __init__(self, length_scale = 1, deviation =1, period = 1):
        self.length_scale = length_scale
        self.deviation = deviation
        self.period = period
        self.printed = False
        
    def covariance(self, X1, X2):
        diff = (X1[:,np.newaxis] - X2)
        square_distance_matrix = np.sum(diff**2,2)
        
        sin_arg = (np.pi*np.sqrt(square_distance_matrix))/self.period
        sin_part = np.sin(sin_arg)**2/(self.length_scale**2)
        
        return self.deviation**2 * np.exp(-2*sin_part)
        
    
    def log_prob(self, X, Y, noise, naive_implementation = False, length_scale = None, deviation = None, period = None):
        if isinstance(length_scale, float):
            self.length_scale = length_scale
        if isinstance(deviation, float):
            self.deviation = deviation
        if isinstance(period, float):
            self.period = period
            
        return self.log_prob_core(X,Y,noise, naive_implementation)
    
    def fit(self, X, Y, noise = 1e-10, init = [1,1,1], bounds = ((1e-5, None), (1e-5, None), (1e-5,None))\
            , method = 'L-BFGS-B'):
        def to_min(parameter):
            try:
                return self.log_prob(X,Y,noise,naive_implementation = False,length_scale = parameter[0],deviation = parameter[1], \
                                 period = parameter[2])
            except Exception as e:
                if not self.printed:
                    print('Using unstable log_prob due to the following issue:',e)
                    self.printed = True
                else:
                    pass
                return self.log_prob(X,Y,noise,naive_implementation = True,length_scale = parameter[0],deviation = parameter[1], \
                                 period = parameter[2])
        res = minimize(to_min, init, bounds=bounds,method=method)
    
    def reset_parameters(self, length_scale, deviation, relative_scaling):
        self.length_scale = length_scale
        self.deviation = deviation
        self.period = period
        
    def __str__(self):
        outstring = ''
        outstring+='Type : \n'
        outstring+='\t Period Kernel \n'
        outstring+='Parameters : \n'
        outstring+='\t length_scale : '+str(self.length_scale)+'\n'
        outstring+='\t deviation : '+str(self.deviation)+'\n'
        outstring+='\t period : '+str(self.period)
        return outstring
        
class GuassianProcessRegression:
    def __init__(self, kernel, noise = 1e-10):
        self.kernel = kernel
        self.noise = noise
        
    def _nan_checker(self, covariance_matrix):
        diag_part = np.diag(covariance_matrix)
        diag_part_copy = diag_part.copy()
        diag_part_copy[diag_part < 0] = 0
        diag_part = diag_part_copy
        std = np.sqrt(diag_part)
        return std
        
    def _posterior(self, new_point, return_cov = False, return_std = False):
        K_XX = self.kernel.covariance(self.X,self.X) + self.noise*np.eye(len(self.X))
        K_sX = self.kernel.covariance(new_point, self.X)
        print(K_sX)
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
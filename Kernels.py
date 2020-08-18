#######################################################
#
# Python script responsible for all of the possible
# kernels being used for the GPs.
#
#######################################################

import numpy as np
import pandas as pd
from scipy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class kernel:
    """
    Base class for the kernels
    """
    def __init__(self):
        pass
    
    def covariance(self, X1, X2):
        """
        The covariance matrix.
        """
        pass
    
    def log_prob(self, X, Y, noise):
        """
        The log liklihood which is used to fit the hyperparameters of the kernels for the GPs.
        """
        pass
    
    def log_prob_core(self, X, Y, noise, naive_implementation = False):
        """
        The core log liklihood.
        """
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
        """
        This function fits the hyperparameters to the data.
        """
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

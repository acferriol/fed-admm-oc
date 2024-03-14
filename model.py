import numpy as np
import collections
from utils import *
from sklearn.kernel_approximation import Nystroem


class Admm_OneClassSVM:
    """Class for the classifier\n
        __init__(method) -> Constructor, set the parameters\n
        fit(method) -> Train the classifier, parameters: X -> dataset, return: best weights,bias and history\n
        decision_function(method) -> Classifier's Decision function\n
        predict(method) -> Predict instance according 
    """
    def __init__(self
                 ,landa=3,
                 max_iter=50,tolMargin=1.5,a1=1.8,a2=1.8,verbose=True,history=collections.defaultdict(list)):
        self.landa    = landa #Regularization parameter
        self.max_iter = max_iter #Max Iterations
        self.tolMargin= tolMargin #Tolerance Parameter
        self.a1       = a1 #Augmented Lagrangian parameter 1
        self.a2       = a2 #Augmented Lagrangian parameter 2
        self.verbose  = verbose #Print or not detaills
        self.history  = history #History of algorithm
        
        self.coeff_   = None #w
        self.offset_  = None #b
        self.dual_coeff = None
        self.dual_offset = None
        

    def decision_function(self,X): #Decision function
        return X @ self.coeff_.ravel() - self.offset_
        
    def predict(self, X): #Predictions
        y = (self.decision_function(X) >= 0).astype(np.int32)
        y[y == 0] = -1  # for consistency with outlier detectors
        return y



def train(model:Admm_OneClassSVM, X,max_iter:int,verbose=False):

    n,m = X.shape
    #mu = 10
    
    #Over-relax parameter
    #fact = 2
    alpha=1.7

    wi = np.random.randn(m,1)
    bi = np.random.randn(1)
    wz = model.coeff_
    bz = model.offset_
    ui = np.zeros((m,1))                # scaled weight multipliers
    vi = np.zeros((1,1))

    
    #Introduce noise
    wi_prev = wi + 0.001*np.random.randn(m,1)*np.sqrt(np.sum(np.square(wi),axis=0))
    bi_prev = bi + 0.001*np.random.randn(1,1)*np.sqrt(np.sum(np.square(bi),axis=0))

    wi,bi = wbi_update(X,wi_prev,bi_prev,wz,ui,model.a1,bz,vi,model.a2,landa=model.landa,n_iter=max_iter)

    oldwz,oldbz = wz,bz
    wir = alpha*wi + (1-alpha)*oldwz
    bir = alpha*bi + (1-alpha)*oldbz

    ui  = ui  + wir - wz
    vi  = vi  + bir - bz
    
    model.coeff_ = wir
    model.offset_ = bir
    model.dual_coeff = ui
    model.dual_offset = vi



def get_model_parameters(model:Admm_OneClassSVM):
    """Returns the paramters of a ADMM_OC model"""
    params = [model.coeff_, model.offset_,model.dual_coeff,model.dual_offset]
    return params

def set_model_parameters(
    model: Admm_OneClassSVM, params
) -> Admm_OneClassSVM:
    """Sets the parameters of a ADMM_OC model"""
    model.coeff_ = params[0]
    model.offset_ = params[1]
    model.coeff_ = params[2]
    model.dual_offset = params[3]
    return model

def set_initial_parameters(
    model: Admm_OneClassSVM,m:int
) -> Admm_OneClassSVM:
    """Sets the parameters of a ADMM_OC model"""
    model.coeff_ = np.random.randn(m,1)
    model.offset_ = np.random.randn(1) 
    model.dual_coeff = np.zeros((m,1))                # scaled weight multipliers
    model.dual_offset = np.zeros((1,1))
    return model
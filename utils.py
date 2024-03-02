from typing import Tuple, Union, List
import numpy as np
import collections


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_model_parameters(model):
    """Returns the parameters model"""
    if model.offset_:
        params = [model.coef_, model.offset_]
    else:
        params = [model.coef_,]
    return params

def set_model_params(
    model , params
):
    """Sets the parameters model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    """
    Sets initial parameters as zeros
    """
    n_nodes = 4 
    n_features = 2 
    



class Admm_OneClassSVM:
    """Class for the classifier\n
        __init__(method) -> Constructor, set the parameters\n
        fit(method) -> Train the classifier, parameters: X -> dataset, return: best weights,bias and history\n
        decision_function(method) -> Classifier's Decision function\n
        predict(method) -> Predict instance according 
    """
    def __init__(self,nodes=4
                 ,landa=3,
                 max_iter=50,tolMargin=1.5,a1=1.8,a2=1.8,verbose=True,history=collections.defaultdict(list),SHOW_SGD=False,title="Report"):
        self.nodes    = nodes #Number of nodes
        self.landa    = landa #Regularization parameter
        self.max_iter = max_iter #Max Iterations
        self.tolMargin= tolMargin #Tolerance Parameter
        self.a1       = a1 #Augmented Lagrangian parameter 1
        self.a2       = a2 #Augmented Lagrangian parameter 2
        self.verbose  = verbose #Print or not detaills
        self.history  = history #History of algorithm
        self.SHOW_SGD = SHOW_SGD #Show wbi-update gradient
        self.report_title = title #Report-picture-title
        
        self.coeff_   = None #w
        self.offset_  = None #b

        
    def fit(self, X,y=None): #Training
        (self.coeff_,self.offset_),(_,_),self.history = ADMMocSVM(
                        X, 
                        J=self.nodes, 
                        landa=self.landa,
                        max_iter = self.max_iter,
                        a1       = self.a1,
                        a2       = self.a2,
                        tolMargin= self.tolMargin,
                        verbose  = self.verbose,
                        history  = self.history,
                        report_title=self.report_title,
                        SHOW_SGD = self.SHOW_SGD
                        )
        return self
        

    def decision_function(self,X): #Decision function
        return X @ self.coeff_.ravel() - self.offset_
        
    def predict(self, X): #Predictions
        y = (self.decision_function(X) >= 0).astype(np.int32)
        y[y == 0] = -1  # for consistency with outlier detectors
        return y
    

class AdamOptimizer:
    def __init__(self, lr1=0.05,lr2=0.05, beta1=0.9, beta2=0.999, epsilon=1e-3):
        self.lr1 = lr1
        self.lr2 = lr2
        self.beta1 = beta1
        self.beta11 = beta1
        self.beta2 = beta2
        self.beta22 = beta2

        self.epsilon = epsilon
        
        self.m1 = None
        self.v1 = None
        self.m2 = None
        self.v2 = None
        self.t = 0
    
    def update(self, w, b, wg, bg):
        if self.m1 is None:
            self.m1 = np.zeros_like(w)
            self.v1 = np.zeros_like(w)
            self.m2 = np.zeros_like(b)
            self.v2 = np.zeros_like(b)
            
        
        self.t += 1
        #Update w
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * wg
        self.v1 = self.beta2 * self.v1 + (1 - self.beta2) * (wg**2)
        
        #Update b
        self.m2 = self.beta11 * self.m2 + (1 - self.beta11) * bg
        self.v2 = self.beta22 * self.v2 + (1 - self.beta22) * (bg**2)

        #step w
        m_hatw = self.m1 / (1 - self.beta1**self.t)
        v_hatw = self.v1 / (1 - self.beta2**self.t)

        #step b
        m_hatb = self.m2 / (1 - self.beta11**self.t)
        v_hatb = self.v2 / (1 - self.beta22**self.t)

        lrw = self.lr1*np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        lrb = self.lr2*np.sqrt(1 - self.beta22**self.t) / (1 - self.beta11**self.t)

        w -= lrw * m_hatw / (np.sqrt(v_hatw) + self.epsilon)
        b -= lrb * m_hatb / (np.sqrt(v_hatb) + self.epsilon)
 
        return w,b
    

def generate_oc_data(N=300, N_outliers=20, 
                     grid_size=(50,50), random_state=None):

    rng = np.random.RandomState(random_state)
    
    #######################
    # Generate train data
    X = 0.3 * rng.randn(N, 2)
    X_train = np.r_[X[0::2,:] + 2, X[1::2,:] - 2]
    y_train = 1 + np.zeros((N))
    
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(N_outliers, 2))
    y_outlier = -1 + np.zeros((N_outliers))

    X_train = np.r_[X_train,X_outliers] # train + outliers
    y_train = np.r_[y_train,y_outlier] #train + outliers labels
    
    print(X_train.shape)
    print(y_train.shape)
    xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 100), np.linspace(-4.5, 4.5, 100))

    return(X_train,y_train, xx, yy)
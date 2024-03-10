from typing import Tuple, Union, List
import numpy as np
import collections

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
    

def wbi_loss(X,wi,bi,wz,ui,a1,bz,vi,a2,landa): #loss w
    """Loss for Augmented Lagrangian\n
        X -> Data\n
        wi,bi -> Primal variables\n
        ui,vi -> Dual variables\n
        wz,bz -> Consensus variables\n
        a1,a2 -> Penalty of Augmented Lagrangian\n
        landa -> Regualrization Parameter\n

        Return a float with Loss Value
    """
    return 2.0*landa*np.sum(np.maximum(0,bi-np.dot(X,wi))) + 0.5*a1*np.sum(np.square(wi-wz+ui)) + 0.5*a2*np.sum(np.square(bi-bz+vi))

def wbi_grad(X,wi,bi,wz,ui,a1,bz,vi,a2,landa): #gradient w,b
    """Gradient of Augmented Lagrangian\n
        X -> Data\n
        wi,bi -> Primal variables\n
        ui,vi -> Dual variables\n
        wz,bz -> Consensus variables\n
        a1,a2 -> Penalty of Augmented Lagrangian\n
        landa -> Regualrization Parameter\n

        Return a float with Gradient Value
    """
    hinge_loss = np.maximum(0,bi-np.dot(X,wi))
    which = hinge_loss > 0
    nw = a1*(wi-wz+ui)
    nb = a2*(bi-bz+vi)
    nw -= 2.0*landa*np.sum(X*which,axis=0,keepdims=True).T
    nb += 2.0*landa*np.sum(which)
    return nw,nb

def wbi_update(X,wi,bi,wz,ui,a1,bz,vi,a2, \
           learn_rate=0.05, \
           n_iter=250,verbose=False,landa=0.02): #adam optimizer for update wi,bi for each node
    """Update wi,bi for a node with Adam Optimizer(Tensorflow)\n
        X -> Data\n
        wi,bi -> Primal variables\n
        ui,vi -> Dual variables\n
        wz,bz -> Consensus variables\n
        a1,a2 -> Penalty of Augmented Lagrangian\n
        n_iter -> Maximun Number of Iterations for Adam, default=200\n
        verbose -> True for shows details about the alghorithm, False otherwise\n
        landa -> Regualrization Parameter\n

        Return new values for wi,bi and history loss for actual update
    """

    N,m = X.shape
    adam = AdamOptimizer()

    history_loss = []

    w = np.random.RandomState(31).randn(m,1)
    b = np.random.RandomState(31).randn(1,1)
    #print(w,b)
    for i in range(n_iter):
    #print(w,b)
        grad = wbi_grad(X,w,b,wz,ui,a1,bz,vi,a2,landa)
        #print(grad)
        w,b = adam.update(w,b,grad[0],grad[1])
        history_loss.append(wbi_loss(X,w,b,wz,ui,a1,bz,vi,a2,landa))
        #print(history_loss[-1])
   
        
    return w,b,history_loss


###########Losses##################
def full_loss(X,w,b,landa): #Completely loss (All dataset) objective function value
    """Completely loss (All dataset) objective function value\n
        w -> consensus w\n
        b -> consensus b\n
        landa -> Regularization parameter\n 

        Return the loss with actual w,b values
    """
    return ((np.sum(np.square(w)) - b) + 2.0*landa*np.sum(np.maximum(0,b-np.dot(X,w)))).flatten()

def admm_loss(A,landa,wi,bi,wz,bz): #Admm loss (each partition)
    """Admm loss (each partition)\n
        A -> Partitions list\n
        wi -> w value for each node(i)\n
        bi -> b value for each node(i)\n
        wz -> consensus w\n
        bz -> consensus b\n
        landa -> Regularization parameter\n 

        Return the sum of losses for each node
    """
    admmLoss = (np.sum(np.square(wz)) - bz)
    res = []
    for i, Ai in enumerate(A):
        temp = bi[:,i]
        res.append(sum(np.maximum(0,temp- Ai @ wi[:,[i]])))
    return (admmLoss+2*landa*sum(res)).ravel()


def generate_data():
    random_state = None

    rng = np.random.RandomState(random_state)
    N = rng.choice([i for i in range(400,1000)],size = 1)[0]
    N_outliers = rng.choice([i for i in range(20,80)],size = 1)[0]
    X = 0.3 * rng.randn(N, 2)
    X_train = np.r_[X[0::2,:] + 2, X[1::2,:] - 2]
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(N_outliers, 2))
    X_train = np.r_[X_train,X_outliers] # train + outliers
    np.random.shuffle(X_train)
    return X_train


def get_test_data():
    random_state = None

    rng = np.random.RandomState(random_state)
    N = rng.choice([i for i in range(400,1000)],size = 1)[0]
    N_outliers = rng.choice([i for i in range(20,80)],size = 1)[0]
    X = 0.3 * rng.randn(N, 2)
    X_train = np.r_[X[0::2,:] + 2, X[1::2,:] - 2]
    y_train = 1 + np.zeros((N))
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(N_outliers, 2))
    y_outlier = -1 + np.zeros((N_outliers))
    X_train = np.r_[X_train,X_outliers] # train + outliers
    y_train = np.r_[y_train,y_outlier]
    return X_train, y_train
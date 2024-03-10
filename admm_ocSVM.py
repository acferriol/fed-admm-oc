"""ADMM-OCSVM is an alghoritm who tried to implements OCSVM with a distributed approach using consensus ADMM"""

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import collections
import time as time

#Graphics
from admm_ocSVM_utils import report,sgd_learning_curve

processors = mp.cpu_count() #Ideal number of nodes

###########FUNCTIONS#############

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
           n_iter=250,verbose=True,landa=0.02): #adam optimizer for update wi,bi for each node
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


##########Partition Data###########
def partition_data(A,nodes,shuffle=False): #Generate partitions for each node. Assume n >> nodes
    """Generate partitions for each node. Assume n >> nodes\n
        A -> Complete dataset\n
        nodes -> Number of nodes for split\n
        shuffle -> True for shuffle the dataset, False otherwise\n

        Return a list with the partitions datasets
    """
    if shuffle:
        np.random.shuffle(A)
    n = A.shape[0]
    jump = n//nodes +1
    lf = 0
    rg = min(n,jump)
    parts = []
    for i in range(0,nodes-1):
        parts.append(A[lf:rg])
        lf += jump
        rg = lf+jump
    parts.append(A[lf:n])      
    return parts


def ADMMocSVM(X, J = 4,
                 landa= .02,
                 max_iter= 50,
                 a1 = 1.8,
                 a2 = 1.8,
                 ABSTOL = 1e-3,
                 RELTOL = 1e-2,
                 tolMargin = 1.,
                 verbose = True,
                 SHOW_SGD = True,
                 report_title = "Report",
                 history = {}):
    """Fit function for ADMM-OCSVM\n
        X -> Dataset\n
        J -> Number of nodes\n
        max_iter -> Maximun number of iterations for ADMM-consensus\n
        a1,a2 -> Penalty parameters of the ADMM Augmented Lagrangian\n
        ABSTOL,RELTOL -> Tolerances for epsilons calculate\n
        tolMargin -> Margin for stop conditions\n
        verbose -> True for shows details about the alghorithm, False otherwise\n
        SHOW_SGD -> True for shows details about the updates, False otherwise\n
        report_title -> Title for the graphics\n
        history -> Save the history of the alghorithm\n

        Return best consensus variables, last consensus variables and history
    """

    n,m = X.shape #instances, dimension
    if verbose:
        print("Instances:",n)
        print("Dimension:",m)    

    #alpha's var parameters
    mu = 10
    fact = 2

    #Over-relax parameter
    alpha=1.7

    #Initialize parameters
    wi = np.random.randn(m,J)           # weights, w_i
    bi = np.random.randn(1,J)           # biases, b_i
    wz  = wi.mean(axis=1,keepdims=True)  # consensus weight, w
    bz = bi.mean(1)                     # consensus b
    ui  = np.zeros((m,J))                # scaled weight multipliers
    vi = np.zeros((1,J))
    sgd_loss = []
    sgd_loss_i = None

    bestloss = np.inf
    best = None
    history['Nodes'].append(J)
    
    history['best_iter'].append(0)

    A = partition_data(X,J,True)

    if verbose:
        print("%+15s"*9%('iter', 'r_norm', 'eps_pri', 's_norm', 'eps_dual','admm_obj' ,'full_obj', 'alpha1','alpha2') )

    for k in range(max_iter):
        
        
        #Introduce noise
        wi_prev = wi + 0.001*np.random.randn(m,J)*np.sqrt(np.sum(np.square(wi),axis=0))
        bi_prev = bi + 0.001*np.random.randn(1,J)*np.sqrt(np.sum(np.square(bi),axis=0))

        ####Update per node wi and bi#### PYSPARK!!!
        for i in range(J):
            wi[:,[i]],bi[:,[i]],sgd_loss_i = wbi_update(A[i],wi_prev[:,[i]],bi_prev[:,[i]],wz,ui[:,[i]],a1,bz,vi[:,[i]],a2,verbose=SHOW_SGD,landa=landa)
            sgd_loss.append(min(sgd_loss_i))
             



        ####Update consensus wz and bz####
        #Overrelax
        oldwz,oldbz = wz,bz
        wir = alpha*wi + (1-alpha)*oldwz
        bir = alpha*bi + (1-alpha)*oldbz

        #update consensus
        wz = J*a1*(wir.mean(1,keepdims=True) + ui.mean(1,keepdims=True)) / (1 + J*a1) 
        bz = bir.mean(1,keepdims=True) + vi.mean(1,keepdims=True) + 1/J*a2

        ####Update dual variables####
        ui  = ui  + wir - wz
        vi  = vi  + bir - bz

        
        
        ####Means####
        #w_mean = wi.mean(axis=1,keepdims=True)
        #b_mean = bi.mean(1)

        ####Full_Loss####
        k_loss = full_loss(X,wz,bz,landa)
        if k_loss < bestloss:
            best = (wz,bz)
            bestloss = k_loss 
            history['best_iter'][0] = k

        solution    = np.concatenate([wi,bi])
        #means       = np.vstack([w_mean,b_mean])
        #meansold    = np.vstack([old_wmean,old_bmean])
        consensusold= np.vstack([oldwz,oldbz])
        consensus   = np.vstack([wz,bz])
        multipliers = 0.5*(a1+a2)*np.concatenate([ui,vi])

        ####Compute residuals and save history####

        ####Residuals####
        p_residual  = np.linalg.norm(solution - consensus) #Primal residual
        d_residual  = np.linalg.norm(-0.5*(a1+a2)*(consensus - consensusold)) #Dual residual

        ####History####
        history['full_Obj'].append( k_loss )
        history['admm_Obj'].append( admm_loss(A,landa,wi,bi,wz,bz) )
        history['r_nor'].append( p_residual )  
        history['s_norm'].append( d_residual )  
        history['a1'].append(a1)
        history['a2'].append(a2)
        ####Epsilons####
        aux = np.sqrt(m)*ABSTOL
        history['eps_pri'].append(  aux + RELTOL*max( np.linalg.norm(solution), np.sqrt(J)*np.linalg.norm(consensus)))
        history['eps_dual'].append( aux + RELTOL*np.linalg.norm(multipliers))

        if verbose:
            print(("%15.0f"+"%15.3f"*8)%(k,
                                                          history['r_nor'][-1], 
                                                          history['eps_pri'][-1], 
                                                          history['s_norm'][-1], 
                                                          history['eps_dual'][-1], 
                                                          history['admm_Obj'][-1], 
                                                          history['full_Obj'][-1],
                                                          history['a1'][-1],
                                                          history['a2'][-1]                  ))
        
        
        ####No improvement check####
        last_ten = history["full_Obj"][-10:]
        if min(last_ten) == last_ten[0] and k>10:
            #print("No improvement")
            break

        
        ####Stop Check####
        if (history['r_nor'][-1]  < tolMargin*history['eps_pri'][-1] and 
            history['s_norm'][-1] < tolMargin*history['eps_dual'][-1] ):
            break
        
        #alphas update
        if p_residual>mu*d_residual:
            a1,a2 = a1*fact,a2*fact
        elif d_residual>mu*p_residual:
            a1,a2 = a1/fact,a2/fact
    

    if verbose:
        #report(4,history,report_title)
        sgd_learning_curve(sgd_loss,"LossSGD",J) 
    #print(best,(wz,bz))
    return best,(wz,bz),history            


######Classifier##########
class Admm_OneClassSVM:
    """Class for the classifier\n
        __init__(method) -> Constructor, set the parameters\n
        fit(method) -> Train the classifier, parameters: X -> dataset, return: best weights,bias and history\n
        decision_function(method) -> Classifier's Decision function\n
        predict(method) -> Predict instance according 
    """
    def __init__(self,nodes=processors
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
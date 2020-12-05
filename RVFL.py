import numpy as np

# IMPLEMENTATION OF COMMON ACTIVATION FUNCTIONS FOR RVFL

def sigmoid(x):
  x = 1/(1+np.exp(-x))
  return x

def radbas(x):
  x = np.exp(-x*x)
  return x

def tribas(x):
  x = np.clip(1.0 - np.fabs(x), 0.0, 1.0)
  return x

def gaussian(x):
  x = np.exp(-pow(x, 2.0)) 
  return x

# MAIN CLASS FOR RVFL 
class RVFL:
  def __init__(self,w=None,b=None,beta=None, hidden_nodes= None):
        self.w = w
        self.b = b
        self.beta = beta
        self.hidden_nodes = hidden_nodes

  # FUNCTION THAT RANDOMLY INITIALIZE WEIGHTS AND BIASES BETWEEN INPUT AND HIDDEN LAYER  
  def initialize_random(self,feature_size, hidden_nodes):
    self.hidden_nodes = hidden_nodes
    self.w = np.random.uniform(low = -1, high = 1,size = (feature_size,self.hidden_nodes))
    self.b = np.random.uniform(size = (self.hidden_nodes,1))
    self.beta = np.random.uniform((self.hidden_nodes,1))
  
  # FUNCTION THAT CALCULATES WEIGHTS BETWEEN HIDDEN-OUTPUT AND INPUT-OUTPUT LAYER USING RIDGE REGRESSION
  def fit(self,data,labels, regularization_param):
    H = np.array((sigmoid(np.dot(data,self.w)+self.b.T)));
    H = np.hstack((H,data))
    size = len(H[0])
    inverse = np.linalg.pinv((H.T@H) + regularization_param*np.identity(size))
    inverse = np.dot(inverse,H.T)
    self.beta = np.dot(inverse,labels)
    
    #FUNCTION TO CALCULATE ACCURACY OF RVFL
  def evaluate(self,predictions,labels):
    acc=0
    for i in range(len(predictions)):
      if(predictions[i]==labels[i]):
        acc+=1
    acc=acc/len(predictions)
    return acc
  
  # FUNCTION TO PREDICT OUTCOME FOR DATASET 
  def predict(self,data,threshold=0.5):
    k = np.dot(data,self.w)+self.b.T
    l = len(k[0])
    div = np.array(self.beta[:l])
    div1 = np.array(self.beta[l:])
    out2 = np.dot(sigmoid(np.dot(data,self.w)+self.b.transpose()),div)
    out1 = np.dot(data,div1)
    out = sigmoid(out1+out2)
    ones = 0
    zero = 0
    for i in range(np.shape(out)[0]):
      if out[i]>threshold:
        out[i]=1
        ones+=1
      else:
        out[i]=0
        zero+=1
    return out
# Code for Extreme Learning Machine

# 3 Activation functions  - sigmoid, tribas and gaussian
def sigmoid(x):
  x = 1/(1+np.exp(-x))
  return x

def tribas(x):
  x = np.clip(1.0 - np.fabs(x), 0.0, 1.0)
  return x

def gaussian(x):
  x = np.exp(-pow(x, 2.0)) 
  return x


class ELM:
  def __init__(self,w=None,b=None,beta=None, hidden_nodes=80):
        self.w = w
        self.b = b
        self.beta = beta
        self.hidden_nodes = hidden_nodes

  def initialize_random(self,feature_size):
    # randomly assign the weights and bias between input layer and hidden layer
    self.w = np.random.uniform(low = -1, high = 1,size = (feature_size,self.hidden_nodes))
    self.b = np.random.uniform(size = (self.hidden_nodes,1))
    self.beta = np.random.uniform((self.hidden_nodes,1))
  
  def fit(self,data,labels):
    # analytically calculate weights between hidden layer and output layer using Moore-Penrose inverse
    inverse = np.linalg.pinv((sigmoid(np.dot(data,self.w)+self.b.T)))
    self.beta = np.dot(inverse,labels)

  def evaluate(self,predictions,labels):
    # evaluate the predictions of ELM and return the AUC-ROC score
    return roc_auc_score(labels,predictions)

  def predict(self,data,threshold):
    # predict output of ELM for the given data using the threshold value
    out= sigmoid(np.dot(sigmoid(np.dot(data,self.w)+self.b.transpose()),self.beta))
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


clf = ELM()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test, 0.5)
clf.evaluate(y_pred,Y_test)

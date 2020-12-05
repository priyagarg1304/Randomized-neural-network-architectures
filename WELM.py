# WELM

# ACTIVATION FUNCTION FOR WELM 
def sigmoid(x):
  x = 1/(1+np.exp(-x))
  return x


class WELM:
  def __init__(self,W = None,w=None,b=None,beta=None, hidden_nodes = 50):
        self.w = w
        self.b = b
        self.beta = beta
        self.W = W 
        self.hidden_nodes = hidden_nodes

  def initialize_random(self,feature_size,no_of_samples):
    self.w = np.random.uniform(low = -1, high = 1,size = (feature_size,self.hidden_nodes))
    self.b = np.random.uniform(size = (self.hidden_nodes,1))
    self.beta = np.random.uniform((self.hidden_nodes,1))
    self.W = np.zeros(shape = (no_of_samples,no_of_samples)) # Weight matrix for sample weightage

# Training the classifier
  def fit(self,data,labels):
    # Creating the weight matrix according to the minority and majority class
    count = [0,0]
    for sample in labels:
      if(sample[0] == 1):
        count[1]+=1
      else: 
        count[0]+=1
    minority = 1
    if(count[1]>count[0]):
      minority = 0
    for sample in range(len(labels)):
      if(labels[sample,0] == minority):
        self.W[sample,sample] = 0.618/count[minority]
        # 0.618 is the golden ratio 
      else: 
        self.W[sample,sample] = 1/count[1-minority]

    # calcultaing H matrix and beta - final weights from hidden to output layer
    H = sigmoid(np.dot(data,self.w)+self.b.T)
    inverse = np.linalg.pinv((np.dot(H.T,np.dot(self.W,H))))
    self.beta = np.dot(inverse,np.dot(H.T,np.dot(self.W,labels)))
    
  #Return accuracy for the predicted values
  def evaluate(self,predictions,labels):
    acc=0
    for i in range(len(predictions)):
      if(predictions[i]==labels[i]):
        acc+=1
    acc=acc/len(predictions)
    return acc

  # Using beta matrix to evaluate the final expected output
  def predict(self,data, threshold):
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


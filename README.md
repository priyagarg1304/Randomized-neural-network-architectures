# Randomized-neural-network-architectures (RNNAs)

## Random Vector Functional Link Network - RVFL

RVFL is a RNNA introduced in 1994. It consists of an input layer, hidden layer and an output layer. The input layer has additional direct links with output layer. The weights and biases between input and layer layer are randomly initialized while those between input to output and hidden to output are calculated analytically. Given below is the structure of RVFL.

![RVFL Structure](/Images/rvfl.png)

The following code can be used for RVFL classifier. 
``` python
from RVFL import RVFL
model = RVFL()
model.initialize_random(feature_size=20, hidden_nodes=100)
model.fit(train_data, train_label, regularization_param=0.0312)
predictions = model.predict(test_data, threshold =0.5)
accuracy = model.evaluate(predictions, test_label)
```

## Extreme Learning Machine - ELM

ELM is a RNNA introduced in 2006. Its structure is similar to a feed-forward neural network with 1 input and 1 output layer and 1 or more hidden layers. The difference between RVFL and ELM is that ELM doesnot have direct links from input to output layer. The weights from the input layer to the output layer are randomly initialized and never changed during to whole training and testing process. The weights between hidden and output layer are caluclated analytically using Moore-Penrose Inverse. Given below is the structure of ELM.

![ELM Structure](/Images/BASICELM%20(1).png)


The following code can be used for ELM classifier. 
``` python
from ELM import ELM
model = ELM()
model.initialize_random(feature_size=20, hidden_nodes=100)
model.fit(train_data, train_label)
predictions = model.predict(test_data, threshold = 0.5)
accuracy = model.evaluate(predictions, test_label)
```

## Weighted Extreme Learning Machine - WELM

WELM is a special kind of RNNA which handles class imbalance using an additional weight matrix.  Its structure is similar to a feed-forward neural network with 1 input and 1 output layer and 1 or more hidden layers. The weights from the input layer to the output layer are randomly initialized and never changed during to whole training and testing process. The weights between hidden and output layer are caluclated analytically using Moore-Penrose Inverse. Given below is the structure of WELM.

![WELM Structure](/Images/welm%20(2).png)


The following code can be used for WELM classifier. 
``` python
from WELM import WELM
model = WELM()
model.initialize_random(feature_size=20, hidden_nodes=100)
model.fit(train_data, train_label)
predictions = model.predict(test_data, threshold = 0.5)
accuracy = model.evaluate(predictions, test_label)
```


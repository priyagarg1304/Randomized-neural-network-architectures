# Randomized-neural-network-architectures (RNNAs)


## Extreme Learning Machine - ELM

## Random Vector Functional Link Network - RVFL

RVFL is a RNNA introduced in 1994. It consists of an input layer, hidden layer and an output layer. The input layer has additional direct links with output layer. The weights and biases between input and layer layer are randomly initialized while those between input to output and hidden to output are calculated analytically. Given below is the structure of RVFL.

![RVFL Structure](/Images/rvfl.png)

The following code can be used for RVFL classifier. 
``` python
from RVFL import RVFL
model = RVFL()
model.initialize_random(feature_size=20, hidden_nodes=100)
model.fit(train_data, train_label, regularization_param=0.0312)
predictions = model.predict(test_data,0.5)
model.evaluate(predictions, test_label)
```

## Weighted Extreme Learning Machine - WELM

WELM is a special kind of RNNA which handles class imbalance using an additional weight matrix. 

# CART-Decision-Tree-And-Naive-Bayes-Example
An example of the CART decision tree and Naive Bayes data mining methods on the breast-cancer.arff dataset from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer)

# Requirements
```
pandas==1.2.1
matplotlib==3.3.4
scipy==1.6.0
numpy==1.20.0
pydotplus==2.0.2
six==1.15.0
ipython==7.20.0
rich==9.11.0
scikit_learn==0.24.1
```

# Graphviz
Graphviz is used to export image of decision tree.
To install download Graphviz and add to path variable. Reboot may be required.

## Naive Bayes

| Confusion matrix        | Actual Positive (1)           | Actual Negative (0)  |
| ------------- |:-------------:| -----:|
| Predicted Positive (1)      | 39 | 9 |
| Predicted Negative (0)      | 9      |   13 |

Accuracy: 74.2%

![ROC curve Naive Bayes](https://raw.githubusercontent.com/Adilius/CART-Decision-Tree-And-Naive-Bayes-Example/main/ROC_NB.png)

## Decision Tree

| Decision Tree        | Actual Positive (1)           | Actual Negative (0)  |
| ------------- |:-------------:| -----:|
| Predicted Positive (1)      | 42 | 6 |
| Predicted Negative (0)      | 15    |  7 |

Accuracy: 70.0%

![ROC curve decision tree](https://raw.githubusercontent.com/Adilius/CART-Decision-Tree-And-Naive-Bayes-Example/main/ROC_DT.PNG)


![Decision Tree](https://raw.githubusercontent.com/Adilius/CART-Decision-Tree-And-Naive-Bayes-Example/main/big_tree.png)
